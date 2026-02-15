# ============================================================
# 0) INSTALL DEPENDENCIES
# ============================================================
!pip install -q transformers accelerate datasets sentencepiece opencv-python-headless

import os, cv2, json, torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
from torch.optim import AdamW
from math import floor

os.environ["WANDB_DISABLED"] = "true"   # disable W&B


# ============================================================
# 1) PATHS & BASIC CONFIG
# ============================================================
VIDEO_DIR    = "/content/videos_extracted/downloaded_videos_new"
CAPTION_JSON = "/content/summaries_new.json"

MAX_VIDEOS         = 170   # how many videos to use (you have ~173–180)
FRAMES_PER_VIDEO   = 15    # final frames per video
CANDIDATES_PER_VID = 60    # how many candidate frames we examine before selecting FRAMES_PER_VIDEO

TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1  # remaining

print("VIDEO_DIR contents (first 10):")
print(os.listdir(VIDEO_DIR)[:10])


# ============================================================
# 2) LOAD CAPTION METADATA & SPLIT BY VIDEO (train/val/test)
# ============================================================
print("\nLoading caption JSON...")
with open(CAPTION_JSON, "r") as f:
    caption_data = json.load(f)

available_videos = {
    os.path.splitext(f)[0] for f in os.listdir(VIDEO_DIR)
    if f.lower().endswith(".mp4")
}
print("Total .mp4 files in VIDEO_DIR:", len(available_videos))

# Keep only videos that actually exist
valid_items = [item for item in caption_data if item["video_id"] in available_videos]

if len(valid_items) == 0:
    raise RuntimeError("No overlap between CAPTION_JSON video_ids and VIDEO_DIR .mp4 files.")

# Limit to MAX_VIDEOS
valid_items = valid_items[:MAX_VIDEOS]
n_total = len(valid_items)
print(f"Usable videos (after filtering & MAX_VIDEOS): {n_total}")

# --- Train/Val/Test split by video index ---
n_train = floor(n_total * TRAIN_RATIO)
n_val   = floor(n_total * VAL_RATIO)
n_test  = n_total - n_train - n_val  # whatever remains

train_items = valid_items[:n_train]
val_items   = valid_items[n_train:n_train+n_val]
test_items  = valid_items[n_train+n_val:]

print(f"Train videos: {len(train_items)}")
print(f"Val   videos: {len(val_items)}")
print(f"Test  videos: {len(test_items)}\n")


# ============================================================
# 3) LOAD BLIP (PRETRAINED) FOR BOTH EMBEDDINGS & TRAINING
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.to(device)


# ============================================================
# 4) HELPER: GET FRAME EMBEDDINGS (VISION ENCODER)
# ============================================================
@torch.no_grad()
def get_frame_embeddings(frames):
    """
    frames: list of PIL Images
    returns: tensor of shape (num_frames, hidden_dim)
    """
    model.eval()
    inputs = processor(images=frames, return_tensors="pt").to(device)
    vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
    # Use pooled CLS embedding as frame embedding
    embs = vision_outputs.pooler_output          # (B, hidden_size)
    embs = F.normalize(embs, dim=1)              # cosine-friendly
    return embs.cpu()                            # keep CPU to be safe


def select_diverse_indices(embs, k):
    """
    Basic k-center greedy selection using cosine similarity.
    embs: tensor (N, D) already L2-normalized
    returns: list of k frame indices
    """
    N = embs.shape[0]
    if N <= k:
        return list(range(N))

    # start from middle frame
    selected = [N // 2]

    # Greedily add frames that are most dissimilar to current set
    while len(selected) < k:
        sims = torch.matmul(embs, embs[selected].T)  # (N, len(selected))
        max_sim, _ = sims.max(dim=1)                # similarity to closest selected

        # avoid reselecting already chosen indices
        max_sim[selected] = 1.0

        # pick the one that is least similar to current set
        next_idx = torch.argmin(max_sim).item()
        selected.append(next_idx)

    selected = sorted(selected)
    return selected


def build_frames_for_split(items, split_name):
    """
    items: list of dicts from caption_data (each has 'video_id' and 'summary')
    split_name: 'train'/'val'/'test' just for logging
    returns: (frames, captions)
    """
    frames = []
    captions = []

    print(f"\n>>> Building {split_name} frames with cosine diversity selection...")

    for item in items:
        vid = item["video_id"]
        full_caption = item["summary"]
        # (Optional) shorten caption if extremely long – keep first sentence.
        caption = full_caption.split(".")[0].strip() + "." if "." in full_caption else full_caption

        video_path = os.path.join(VIDEO_DIR, vid + ".mp4")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"[{split_name}][SKIP] Could not open video: {vid}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"[{split_name}][SKIP] No frames in video: {vid}")
            cap.release()
            continue

        # Uniformly sample up to CANDIDATES_PER_VID frames
        step = max(1, total_frames // CANDIDATES_PER_VID)
        candidate_frames = []
        frame_id = 0

        while frame_id < total_frames and len(candidate_frames) < CANDIDATES_PER_VID:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            candidate_frames.append(Image.fromarray(frame_rgb))
            frame_id += step

        cap.release()

        if len(candidate_frames) == 0:
            print(f"[{split_name}][SKIP] No candidate frames from video: {vid}")
            continue

        print(f"[{split_name}] Video {vid}: total_frames={total_frames}, "
              f"candidates={len(candidate_frames)}")

        # Get embeddings and pick FRAMES_PER_VIDEO diverse frames
        embs = get_frame_embeddings(candidate_frames)  # (N, D)
        select_idx = select_diverse_indices(embs, FRAMES_PER_VIDEO)
        print(f"  -> selected {len(select_idx)} frames (cosine-diverse)")

        for idx in select_idx:
            frames.append(candidate_frames[idx])
            captions.append(caption)

    print(f"{split_name} split: total selected frames = {len(frames)}")
    return frames, captions


# ============================================================
# 5) BUILD TRAIN / VAL / TEST FRAMES
# ============================================================
train_frames, train_captions = build_frames_for_split(train_items, "train")
val_frames,   val_captions   = build_frames_for_split(val_items, "val")
test_frames,  test_captions  = build_frames_for_split(test_items, "test")

if len(train_frames) == 0:
    raise RuntimeError("No train frames selected. Check video paths / captions.")

print("\nTotal frames:")
print("  train:", len(train_frames))
print("  val  :", len(val_frames))
print("  test :", len(test_frames))


# ============================================================
# 6) FREEZE EVERYTHING EXCEPT TEXT DECODER
# ============================================================
for name, param in model.named_parameters():
    if name.startswith("text_decoder"):
        param.requires_grad = True
    else:
        param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"\nTrainable params: {trainable} / {total} "
      f"({100.0 * trainable / total:.3f}% — only text decoder)")


# ============================================================
# 7) DATASETS & DATALOADERS
# ============================================================
class FrameDataset(Dataset):
    def __init__(self, frames, captions, processor):
        self.frames = frames
        self.captions = captions
        self.processor = processor

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img = self.frames[idx]
        caption = self.captions[idx]

        inputs = self.processor(
            images=img,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )

        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values":   inputs["pixel_values"][0],
            "input_ids":      input_ids,
            "attention_mask": inputs["attention_mask"][0],
            "labels":         labels
        }

train_dataset = FrameDataset(train_frames, train_captions, processor)
val_dataset   = FrameDataset(val_frames,   val_captions,   processor) if len(val_frames)  > 0 else None
test_dataset  = FrameDataset(test_frames,  test_captions,  processor) if len(test_frames) > 0 else None

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=2, shuffle=False) if val_dataset  else None
test_loader  = DataLoader(test_dataset,  batch_size=2, shuffle=False) if test_dataset else None

print("\nDataset sizes (frames):")
print("  train:", len(train_dataset))
print("  val  :", len(val_dataset)  if val_dataset  else 0)
print("  test :", len(test_dataset) if test_dataset else 0)


# ============================================================
# 8) TRAINING LOOP WITH OPTIONAL VAL LOSS
# ============================================================
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

EPOCHS = 3   # you can increase later
print("\nTraining...")

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0.0
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    for step, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_train_loss += loss.item()

        if step % 20 == 0:
            print(f"  [train] step {step:03d} | loss = {loss.item():.4f}")

    avg_train_loss = total_train_loss / max(1, len(train_loader))
    print(f"Epoch {epoch+1} train loss: {avg_train_loss:.4f}")

    # ---- Validation loss (optional) ----
    if val_loader is not None and len(val_loader) > 0:
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                total_val_loss += outputs.loss.item()
        avg_val_loss = total_val_loss / max(1, len(val_loader))
        print(f"Epoch {epoch+1} val loss  : {avg_val_loss:.4f}")

print("\nTraining completed!")


# ============================================================
# 9) QUICK TEST: CAPTION A FRAME (FROM TEST OR TRAIN)
# ============================================================
print("\nTesting caption generation on a sample frame...")

model.eval()
if test_frames:
    sample_img = test_frames[0]
    sample_gt  = test_captions[0]
else:
    sample_img = train_frames[0]
    sample_gt  = train_captions[0]

inputs = processor(images=sample_img, return_tensors="pt").to(device)
with torch.no_grad():
    gen = model.generate(**inputs, max_new_tokens=30)

print("GT caption      :", sample_gt)
print("Generated       :", processor.decode(gen[0], skip_special_tokens=True))


# ============================================================
# 10) SAVE THE TRAINED MODEL
# ============================================================
SAVE_DIR = "/content/finetuned_blip"

model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)

print("\nModel saved to:", SAVE_DIR)
