# smart_video_summary.py
#architechure III

import torch
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    T5Tokenizer,
    T5ForConditionalGeneration
)
import clip

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIDEO_PATH = "C:/Users/nikhi/Downloads/Lion vs. Wildebeest_ How Lions Hunt as a Pride.mp4"

HIST_THRESHOLD = 0.3
KEYFRAMES_PER_VIDEO = 15
TOP_K_CAPTIONS = 8  # top important captions used for T5 summary


# ----------------------------------------------------
# LOAD MODELS (BLIP-2, T5, CLIP)
# ----------------------------------------------------
print("Loading BLIP-2...")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-350m")
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-350m",
    device_map="auto"
)
blip_model.eval()

print("Loading T5...")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(DEVICE)

print("Loading CLIP...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()


# ----------------------------------------------------
# HISTOGRAM DIFFERENCE
# ----------------------------------------------------
def hist_diff(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    h1 = cv2.calcHist([gray1], [0], None, [256], [0,256])
    h2 = cv2.calcHist([gray2], [0], None, [256], [0,256])
    h1 = cv2.normalize(h1, h1).flatten()
    h2 = cv2.normalize(h2, h2).flatten()
    return cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)


# ----------------------------------------------------
# KEYFRAME EXTRACTION (Scene-change + KMeans)
# ----------------------------------------------------
def extract_smart_keyframes(video_path, num_keyframes=KEYFRAMES_PER_VIDEO):
    cap = cv2.VideoCapture(video_path)
    frames = []
    sampled_images = []
    last_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if last_frame is None or hist_diff(last_frame, frame) > HIST_THRESHOLD:
            frames.append(frame.copy())
            sampled_images.append(
                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            )

        last_frame = frame

    cap.release()

    if len(sampled_images) == 0:
        return []

    # CLIP embeddings for clustering
    emb_list = []

    with torch.no_grad():
        for img in sampled_images:
            inp = clip_preprocess(img).unsqueeze(0).to(DEVICE)
            emb = clip_model.encode_image(inp)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb_list.append(emb.cpu().numpy())

    embs = np.vstack(emb_list)

    n_clusters = min(num_keyframes, len(frames))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embs)

    keyframes = []

    for c in range(n_clusters):
        idxs = np.where(kmeans.labels_ == c)[0]
        center = kmeans.cluster_centers_[c]
        chosen = idxs[np.argmin(np.linalg.norm(embs[idxs] - center, axis=1))]
        keyframes.append(frames[chosen])

    return keyframes


# ----------------------------------------------------
# IMPORTANCE SCORING USING CLIP UNIQUENESS
# ----------------------------------------------------
def compute_importance(frames):
    emb_list = []

    with torch.no_grad():
        for frame in frames:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inp = clip_preprocess(img).unsqueeze(0).to(DEVICE)
            emb = clip_model.encode_image(inp)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb_list.append(emb.cpu().numpy())

    embs = np.vstack(emb_list)
    sim = cosine_similarity(embs)
    uniqueness = 1 - sim.mean(axis=1)

    return uniqueness


# ----------------------------------------------------
# T5 SUMMARIZER (Final Stage)
# ----------------------------------------------------
def summarize_text_with_t5(text, max_len=80):
    inp = "summarize: " + text
    tokens = t5_tokenizer.encode(
        inp, return_tensors="pt", truncation=True
    ).to(DEVICE)

    with torch.no_grad():
        out = t5_model.generate(
            tokens,
            max_length=max_len,
            num_beams=4,
            early_stopping=True
        )

    return t5_tokenizer.decode(out[0], skip_special_tokens=True)


# ----------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------
def generate_summary(video_path):
    print("Extracting keyframes...")
    frames = extract_smart_keyframes(video_path)

    if len(frames) == 0:
        return "No keyframes extracted."

    print("Computing importance scores...")
    importance = compute_importance(frames)

    # Sort by importance
    order = np.argsort(importance)[::-1]
    sorted_frames = [frames[i] for i in order]

    print("Generating BLIP-2 captions...")
    captions = []

    with torch.no_grad():
        for frame in sorted_frames:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = processor(
                images=img,
                text="Describe this video frame.",
                return_tensors="pt"
            ).to(DEVICE)

            out_ids = blip_model.generate(
                **inputs,
                max_length=64,
                num_beams=3
            )

            caption = processor.decode(out_ids[0], skip_special_tokens=True)
            captions.append(caption)

    # Select top-K most important captions
    selected = captions[:TOP_K_CAPTIONS]

    # Remove duplicates
    selected = list(dict.fromkeys(selected))

    # Merge into one text
    long_text = " ".join(selected)

    print("Summarizing with T5...")
    final_summary = summarize_text_with_t5(long_text)

    return final_summary


# ----------------------------------------------------
# RUN
# ----------------------------------------------------
print("\nRunning smart video summarizer...\n")
summary = generate_summary(VIDEO_PATH)

print("\n==============================")
print("FINAL SUMMARY")
print("==============================\n")
print(summary)
