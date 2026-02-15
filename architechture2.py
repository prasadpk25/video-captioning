"""
train_pipeline.py
End-to-end: hybrid keyframe extraction -> CLIP embeddings -> Fine-tune T5 -> evaluate/save
"""

import os
import cv2
import math
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import clip  
from transformers import T5Tokenizer, T5ForConditionalGeneration
from rouge_score import rouge_scorer

# -----------------------
# CONFIG
# -----------------------
VIDEO_DIR = "Datasets\downloaded_videos_new"
CSV_PATH = "Datasets\summaries_new.csv"
EMB_DIR = "embeddings"         
BATCH_SIZE = 4                # smaller if CPU
NUM_WORKERS = 0               # use >0 on Linux/GPU machines
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLUSTERS = 6              # number of clusters for k-means part
KEYFRAMES_PER_VIDEO = 8       # final number of keyframes to use for embedding averaging
T5_MODEL_NAME = "t5-small"
EPOCHS = 30
LR = 5e-5
MAX_TARGET_LENGTH = 64

os.makedirs(EMB_DIR, exist_ok=True)

# -----------------------
# LOAD CLIP
# -----------------------
print("Loading CLIP...")
clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)

# -----------------------
# HYBRID KEYFRAME EXTRACTION
# -----------------------
def extract_scene_keyframes(video_path, bhat_threshold=0.2):
    cap = cv2.VideoCapture(video_path)
    keyframes = []
    last_hist = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0,256])
        hist = cv2.normalize(hist, hist).flatten()
        if last_hist is not None:
            diff = cv2.compareHist(last_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if diff > bhat_threshold:
                keyframes.append(frame.copy())
        last_hist = hist
    cap.release()
    return keyframes

def extract_kmeans_keyframes(video_path, num_clusters=NUM_CLUSTERS, sample_every_n=3):
    cap = cv2.VideoCapture(video_path)
    frames = []
    imgs_for_clip = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # optionally downsample frames to speed embedding
        if i % sample_every_n == 0:
            frames.append(frame.copy())
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgs_for_clip.append(img)
        i += 1
    cap.release()
    if len(imgs_for_clip) == 0:
        return []
    # compute CLIP embeddings in batches
    all_embs = []
    batch = []
    with torch.no_grad():
        for img in imgs_for_clip:
            batch.append(preprocess(img).unsqueeze(0))
            if len(batch) >= 16:
                b = torch.cat(batch, dim=0).to(DEVICE)
                e = clip_model.encode_image(b).cpu().numpy()
                all_embs.append(e)
                batch = []
        if batch:
            b = torch.cat(batch, dim=0).to(DEVICE)
            e = clip_model.encode_image(b).cpu().numpy()
            all_embs.append(e)
    if len(all_embs) == 0:
        return []
    embs = np.concatenate(all_embs, axis=0)
    # cluster and pick representative frame per cluster
    n_clusters = min(num_clusters, embs.shape[0])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embs)
    keyframes = []
    for c in range(n_clusters):
        idxs = np.where(kmeans.labels_ == c)[0]
        center = kmeans.cluster_centers_[c]
        chosen = idxs[np.argmin(np.linalg.norm(embs[idxs] - center, axis=1))]
        keyframes.append(frames[chosen])   # correct index
 # map back to original index approx
    return keyframes

def extract_hybrid_keyframes(video_path, kmeans_clusters=NUM_CLUSTERS):
    scene = extract_scene_keyframes(video_path)
    kmeans = extract_kmeans_keyframes(video_path, num_clusters=kmeans_clusters)
    all_frames = []
    # order: scene then kmeans to ensure scene anchors kept
    for f in scene + kmeans:
        all_frames.append(f)
    # deduplicate by histogram similarity
    unique = []
    hists = []
    for frame in all_frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0,256])
        hist = cv2.normalize(hist, hist).flatten()
        is_dup = False
        for h in hists:
            diff = cv2.compareHist(h, hist, cv2.HISTCMP_BHATTACHARYYA)
            if diff < 0.03:  # nearly identical
                is_dup = True
                break
        if not is_dup:
            unique.append(frame)
            hists.append(hist)
    # if too many frames, downsample to KEYFRAMES_PER_VIDEO evenly
    if len(unique) > KEYFRAMES_PER_VIDEO:
        idxs = np.linspace(0, len(unique)-1, KEYFRAMES_PER_VIDEO).astype(int)
        unique = [unique[i] for i in idxs]
    return unique

# -----------------------
# CLIP embedding per video (saved on disk)
# -----------------------
def video_to_embedding(video_id):
    emb_file = os.path.join(EMB_DIR, f"{video_id}.npy")
    if os.path.exists(emb_file):
        return np.load(emb_file)

    vpath = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
    if not os.path.exists(vpath):
        print(f"Video not found: {vpath}")
        return np.zeros(512)  # fallback embedding for missing video

    frames = extract_hybrid_keyframes(vpath)

    # fallback: uniform sampling if hybrid extraction fails
    if len(frames) == 0:
        cap = cv2.VideoCapture(vpath)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            print(f"No frames found in video: {video_id}")
            return np.zeros(512)  # fallback embedding for empty video
        idxs = np.linspace(0, max(1, total-1), KEYFRAMES_PER_VIDEO).astype(int)
        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, frm = cap.read()
            if ret:
                frames.append(frm)
        cap.release()
    
    if len(frames) == 0:
        print(f"Failed to extract frames from {video_id}")
        return None


    # compute CLIP embeddings (batch)
    imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    feats = []
    with torch.no_grad():
        batch = []
        for img in imgs:
            batch.append(preprocess(img).unsqueeze(0))
            if len(batch) >= 16:
                b = torch.cat(batch).to(DEVICE)
                e = clip_model.encode_image(b).cpu().numpy()
                feats.append(e)
                batch = []
        if batch:
            b = torch.cat(batch).to(DEVICE)
            e = clip_model.encode_image(b).cpu().numpy()
            feats.append(e)

    if len(feats) == 0:
        print(f"No embeddings computed for {video_id}")
        return np.zeros(512)

    feats = np.concatenate(feats, axis=0)
    feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    mean_feat = feats.mean(axis=0)
    np.save(emb_file, mean_feat)
    return mean_feat

# -----------------------
# Load CSV and build embedding dataset
# -----------------------
df = pd.read_csv("Datasets/summaries_new.csv")
print("Total rows:", len(df))

# Create embeddings for all videos (caching)
video_ids = df["video_id"].astype(str).tolist()
summaries = df["summary"].astype(str).tolist()

valid_video_ids = []
valid_summaries = []
emb_list = []

for vid, summ in tqdm(zip(video_ids, summaries), total=len(video_ids), desc="Embedding videos"):
    emb = video_to_embedding(vid)
    if emb is None or np.all(emb == 0):   # check fallback embedding
        print(f"Skipping video {vid} and summary")
        continue
    emb_list.append(torch.tensor(emb, dtype=torch.float32))
    valid_video_ids.append(vid)
    valid_summaries.append(summ)

# save only valid embeddings and CSV
emb_tensor = torch.stack(emb_list)
torch.save(emb_tensor, os.path.join(EMB_DIR, "all_embeddings.pt"))

df_valid = pd.DataFrame({"video_id": valid_video_ids, "summary": valid_summaries})
df_valid.to_csv("Datasets/summaries_valid.csv", index=False)


emb_tensor = torch.stack(emb_list)  # shape (N, D)
print("Embeddings shape:", emb_tensor.shape)
torch.save(emb_tensor, os.path.join(EMB_DIR, "all_embeddings.pt"))

# -----------------------
# Dataset for fine-tuning T5
# -----------------------
tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_NAME)

class VideoSummaryDataset(Dataset):
    def __init__(self, embeddings_tensor, summaries, tokenizer, max_target_len=MAX_TARGET_LENGTH):
        self.emb = embeddings_tensor
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_target_len = max_target_len
    def __len__(self):
        return len(self.summaries)
    def __getitem__(self, idx):
        emb = self.emb[idx]
        tgt = self.summaries[idx]
        tgt_enc = self.tokenizer(tgt, truncation=True, padding="max_length", max_length=self.max_target_len, return_tensors="pt")
        return {
            "embedding": emb,
            "labels": tgt_enc["input_ids"].squeeze(),
            "decoder_attention_mask": tgt_enc["attention_mask"].squeeze()
        }
dataset = VideoSummaryDataset(emb_tensor, valid_summaries, tokenizer)
# simple split
n = len(dataset)
n_train = int(0.9 * n)
train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n - n_train])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# -----------------------
# Model: project CLIP dim -> T5 encoder embeddings then decode
# We'll adapt T5 by feeding the projected embedding as encoder input_embeds
# -----------------------
t5 = T5ForConditionalGeneration.from_pretrained(T5_MODEL_NAME).to(DEVICE)
proj = nn.Linear(emb_tensor.shape[1], t5.config.d_model).to(DEVICE)

# optimizer (only finetune t5 + proj or freeze some layers as desired)
optimizer = torch.optim.AdamW(list(t5.parameters()) + list(proj.parameters()), lr=LR)

# training loop
scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)

def train_one_epoch(epoch):
    t5.train()
    proj.train()
    running_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
        emb = batch["embedding"].to(DEVICE)              # (B, D)
        labels = batch["labels"].to(DEVICE)             # (B, L)
        decoder_mask = batch["decoder_attention_mask"].to(DEVICE)
        # project -> inputs_embeds shape (B, seq_len=1, d_model)
        inputs_embeds = proj(emb).unsqueeze(1)
        # Note: T5 expects encoder_inputs; we pass inputs_embeds and let decoder generate
        outputs = t5(inputs_embeds=inputs_embeds, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()
    print(f"Epoch {epoch+1} train loss: {running_loss/len(train_loader):.4f}")

def evaluate():
    t5.eval()
    proj.eval()
    all_refs = []
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            emb = batch["embedding"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            inputs_embeds = proj(emb).unsqueeze(1)
            # use generate to obtain predictions
            # create a dummy input_ids for encoder? we directly use inputs_embeds
            preds = t5.generate(inputs_embeds=inputs_embeds, max_length=MAX_TARGET_LENGTH, num_beams=2)
            for p, l in zip(preds, labels):
                pred_text = tokenizer.decode(p, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                ref_text = tokenizer.decode(l, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                all_preds.append(pred_text)
                all_refs.append(ref_text)
    # compute ROUGE averaged
    agg = {"rouge1":[], "rouge2":[], "rougeL":[]}
    for ref, pred in zip(all_refs, all_preds):
        scores = scorer.score(ref, pred)
        agg["rouge1"].append(scores["rouge1"].fmeasure)
        agg["rouge2"].append(scores["rouge2"].fmeasure)
        agg["rougeL"].append(scores["rougeL"].fmeasure)
    mean_scores = {k: np.mean(v) for k,v in agg.items()}
    print("Validation ROUGE:", mean_scores)
    return mean_scores

# -----------------------
# Run training
# -----------------------
for epoch in range(EPOCHS):
    train_one_epoch(epoch)
    evaluate()

# save model + proj
os.makedirs("saved_model", exist_ok=True)
t5.save_pretrained("saved_model/t5_model")
tokenizer.save_pretrained("saved_model/t5_tokenizer")
torch.save(proj.state_dict(), "saved_model/proj.pt")
print("Saved model and projection.")
