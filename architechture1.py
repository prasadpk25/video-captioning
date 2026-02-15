# import cv2
# import numpy as np
# import torch
# import torch.nn.functional as F
# from tqdm import tqdm
# from scenedetect import VideoManager, SceneManager
# from scenedetect.detectors import ContentDetector
# from PIL import Image
# import clip
# from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM

# # ----------------------
# # DEVICE
# # ----------------------
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # ----------------------
# # LOAD CLIP
# # ----------------------
# clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
# clip_model = clip_model.to(device)

# # ----------------------
# # SCENE DETECTION
# # ----------------------
# def detect_scenes(video_path):
#     video_manager = VideoManager([video_path])
#     scene_manager = SceneManager()
#     scene_manager.add_detector(ContentDetector(threshold=27.0))
#     video_manager.start()
#     scene_manager.detect_scenes(frame_source=video_manager)
#     scene_list = scene_manager.get_scene_list()
#     scenes_frames = [(int(s[0].get_frames()), int(s[1].get_frames())) for s in scene_list]
#     video_manager.release()
#     return scenes_frames

# # ----------------------
# # FRAME EXTRACTOR
# # ----------------------
# def frame_at(cap, frame_no):
#     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
#     ret, frame = cap.read()
#     if not ret or frame is None:
#         return None
#     return frame

# # ----------------------
# # MOTION-BASED FRAME SELECTOR
# # ----------------------
# def pick_action_frame_in_shot(cap, start_f, end_f, max_checks=10):
#     step = max(1, (end_f - start_f) // max_checks)
#     best_score, best_frame, best_idx = -1, None, start_f

#     for f_idx in range(start_f, end_f, step):
#         f = frame_at(cap, f_idx)
#         if f is None:
#             continue
#         g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
#         nxt = f_idx + step
#         f2 = frame_at(cap, nxt)
#         if f2 is None:
#             continue
#         g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

#         # Prevent shape mismatch
#         if g.ndim != 2 or g2.ndim != 2:
#             continue

#         flow = cv2.calcOpticalFlowFarneback(g, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#         mag = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))

#         if mag > best_score:
#             best_score, best_frame, best_idx = mag, f, f_idx
#     return best_idx, best_frame, best_score

# # ----------------------
# # EXTRACT KEYFRAMES
# # ----------------------
# def extract_keyframes_by_scenes(video_path, top_k_perc=0.2, max_keyframes=None):
#     scenes = detect_scenes(video_path)
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
#     collected = []

#     for (s_f, e_f) in scenes:
#         if e_f <= s_f:
#             continue
#         bfno, bframe, bscore = pick_action_frame_in_shot(cap, s_f, e_f, max_checks=40)
#         if bframe is None:
#             continue
#         rgb = cv2.cvtColor(bframe, cv2.COLOR_BGR2RGB)
#         timestamp_ms = (bfno / fps) * 1000.0
#         collected.append({"frame_no": bfno, "frame": rgb, "score": bscore, "time_ms": timestamp_ms})

#     cap.release()

#     if max_keyframes is None:
#         max_keep = max(1, int(len(collected) * top_k_perc))
#     else:
#         max_keep = max_keyframes

#     collected_sorted = sorted(collected, key=lambda x: x["score"], reverse=True)[:max_keep]
#     return collected_sorted

# # ----------------------
# # CLIP ENCODING
# # ----------------------
# def encode_frames_clip(frames_list, batch_size=32):
#     tensors = []
#     with torch.no_grad():
#         for i in range(0, len(frames_list), batch_size):
#             batch_imgs = frames_list[i:i + batch_size]
#             proc = torch.stack([clip_preprocess(Image.fromarray(img)) for img in batch_imgs]).to(device)
#             feats = clip_model.encode_image(proc)
#             feats = F.normalize(feats, dim=-1)
#             tensors.append(feats.cpu())
#     return torch.cat(tensors, dim=0).to(device)

# # ----------------------
# # BLIP CAPTIONING
# # ----------------------
# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# blip = BlipForConditionalGeneration.from_pretrained(
#     "Salesforce/blip-image-captioning-base",
#     torch_dtype=torch.float32
# ).to(device)

# # ----------------------
# # SUMMARIZER (Small Model)
# # ----------------------
# model_name = "tiiuae/falcon-rw-1b"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token
# lm = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# # ----------------------
# # MAIN PIPELINE
# # ----------------------
# video_path = "/content/video.mp4"
# collected = extract_keyframes_by_scenes(video_path, top_k_perc=0.5, max_keyframes=20)
# frames_rgb = [c["frame"] for c in collected]
# frame_nos = [c["frame_no"] for c in collected]
# scores = np.array([c["score"] for c in collected])

# embs = encode_frames_clip(frames_rgb, batch_size=16)

# sim = embs @ embs.T
# unique = 1 - sim.mean(dim=1).cpu().numpy()
# unique_n = (unique - unique.min()) / (unique.max() - unique.min() + 1e-8)
# motion_n = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

# combined = 0.6 * unique_n + 0.4 * motion_n
# order = np.argsort(combined)[::-1]
# final_frames = [frames_rgb[i] for i in order]
# final_scores = combined[order]
# final_frame_nos = [frame_nos[i] for i in order]

# # ----------------------
# # CAPTIONING
# # ----------------------
# captions = []
# for f in final_frames:
#     inp = processor(Image.fromarray(f), return_tensors="pt").to(device)
#     out = blip.generate(**inp, max_new_tokens=50)
#     captions.append(processor.decode(out[0], skip_special_tokens=True))

# # ----------------------
# # SUMMARY
# # ----------------------
# prompt = "Summarize the following captions into one short paragraph:\n" + "\n".join(captions)
# inputs = tokenizer(prompt, return_tensors="pt").to(lm.device)
# output = lm.generate(**inputs, max_new_tokens=200)
# summary = tokenizer.decode(output[0], skip_special_tokens=True)

# print("\nGenerated Summary:\n", summary)




# import numpy as np
# import torch
# import torch.nn.functional as F
# from PIL import Image
# import matplotlib.pyplot as plt

# # =============================
# # 1️⃣ Compute Frame Importance
# # =============================
# def compute_importance(embs):
#     """
#     Estimate how unique each frame is using cosine similarity.
#     Frames less similar to others are more important.
#     """
#     sim_matrix = torch.matmul(embs, embs.T)
#     sim_scores = sim_matrix.mean(dim=1)
#     importance = 1 - (sim_scores / sim_scores.max())  # lower similarity → more unique
#     return importance.cpu().numpy()

# importance_weights = compute_importance(clip_embs)
# print(f"[INFO] Computed importance for {len(importance_weights)} frames.")

# # =============================
# # 2️⃣ Select Top Key Frames
# # =============================
# def select_keyframes(frames, weights, top_k=10):
#     idxs = np.argsort(weights)[-top_k:]  # top unique frames
#     key_frames = [frames[i] for i in idxs]
#     return key_frames, idxs

# key_frames, key_idxs = select_keyframes(frames, importance_weights, top_k=10)
# print(f"[INFO] Selected {len(key_frames)} key frames.")

# # =============================
# # 3️⃣ Visualize Key Frames
# # =============================
# plt.figure(figsize=(20, 6))
# for i, frame in enumerate(key_frames):
#     plt.subplot(2, 5, i+1)
#     plt.imshow(frame)
#     plt.axis("off")
#     plt.title(f"Key Frame {i+1}")
# plt.show()

# # =============================
# # 4️⃣ Generate Visual Summary
# # =============================
# # Instead of text, we print timestamps or relative positions
# summary_info = [f"Frame {idx} (importance={importance_weights[idx]:.3f})"
#                 for idx in key_idxs]
# print("\n🎬 Visual Summary of Key Moments:\n")
# print("\n".join(summary_info))






# # ==========================================
# # 🎬 CLIP-Based Video Visual Summarizer (No Audio)
# # ==========================================
# import torch
# import torch.nn.functional as F
# import clip
# import cv2
# import numpy as np
# from PIL import Image
# from tqdm import tqdm

# # =============================
# # 1. Load CLIP model
# # =============================

# device = "cuda" if torch.cuda.is_available() else "cpu"
# clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# # =============================
# # 2. Frame extraction
# # =============================
# def extract_frames(video_path, stride=10):
#     cap = cv2.VideoCapture(video_path)
#     frames, times = [], []
#     idx = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if idx % stride == 0:
#             frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#             times.append(cap.get(cv2.CAP_PROP_POS_MSEC))
#         idx += 1
#     cap.release()
#     print(f"[INFO] Extracted {len(frames)} frames.")
#     return frames, times

# # =============================
# # 3. Get CLIP embeddings
# # =============================
# def get_clip_embeddings(frames):
#     embs = []
#     with torch.no_grad():
#         for f in tqdm(frames, desc="Encoding frames with CLIP"):
#             img = clip_preprocess(Image.fromarray(f)).unsqueeze(0).to(device)
#             feat = clip_model.encode_image(img)
#             feat = F.normalize(feat, dim=-1)
#             embs.append(feat)
#     embs = torch.cat(embs, dim=0)
#     print(f"[SUCCESS] Generated embeddings of shape: {embs.shape}")
#     return embs

# # =============================
# # 4. Select key frames (by uniqueness)
# # =============================
# def select_keyframes(frames, embs, top_k=10):
#     sim = embs @ embs.T
#     uniqueness = 1 - sim.mean(dim=1)
#     idx = uniqueness.topk(top_k).indices.cpu().numpy()
#     keyframes = [frames[i] for i in idx]
#     return keyframes, uniqueness[idx], idx

# # =============================
# # 5. Describe each key frame using CLIP text prompts
# # =============================
# def describe_frame(frame):
#     prompt_texts = [
#         "a lion chasing a wildebeest",
#         "a lion attacking its prey",
#         "a lion eating meat",
#         "a lion hunting with its pride",
#         "a wildebeest running away",
#         "a fight between predator and prey",
#         "a calm savannah with animals",
#         "a lion resting after a hunt",
#         "a group of lions watching prey",
#         "a herd of wildebeests moving together"
#     ]

#     with torch.no_grad():
#         text_tokens = clip.tokenize(prompt_texts).to(device)
#         text_embs = clip_model.encode_text(text_tokens)
#         text_embs = F.normalize(text_embs, dim=-1)

#         img = clip_preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
#         img_emb = clip_model.encode_image(img)
#         img_emb = F.normalize(img_emb, dim=-1)

#         sims = (img_emb @ text_embs.T).squeeze(0)
#         best_idx = sims.argmax().item()
#         return prompt_texts[best_idx]

# # =============================
# # 6. Summarize video visually
# # =============================
# def summarize_video(video_path, stride=15, top_k=10):
#     frames, _ = extract_frames(video_path, stride)
#     embs = get_clip_embeddings(frames)
#     keyframes, importance, idx = select_keyframes(frames, embs, top_k)

#     print("\n🎬 Visual Summary of Key Moments:\n")
#     captions = []
#     for i, (f, imp, frame_idx) in enumerate(zip(keyframes, importance, idx)):
#         desc = describe_frame(f)
#         captions.append(desc)
#         print(f"Frame {frame_idx:03d} (importance={imp:.3f}) → {desc}")

#     print("\n🧠 Final Text Summary:")
#     summary = " ".join(sorted(set(captions), key=captions.index))
#     print(summary)
#     return summary

# # =============================
# # 7. Run the pipeline
# # =============================
# video_path = "/content/Lion vs. Wildebeest_ How Lions Hunt as a Pride (1).mp4"
# summary = summarize_video(video_path, stride=15, top_k=10)




import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from PIL import Image
import clip
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    T5Tokenizer,
    T5ForConditionalGeneration
)

# ----------------------
# DEVICE CONFIG
# ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ----------------------
# LOAD CLIP MODEL
# ----------------------
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

# ----------------------
# SCENE DETECTION USING SCENEDETECT
# ----------------------
def detect_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27.0))  # sensible threshold
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    scene_list = scene_manager.get_scene_list()
    frames = [(int(s[0].get_frames()), int(s[1].get_frames())) for s in scene_list]

    video_manager.release()
    return frames

# ----------------------
# FRAME EXTRACTOR
# ----------------------
def frame_at(cap, frame_no):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    return frame if ret else None

# ----------------------
# MOTION-BASED FRAME SELECTOR
# ----------------------
def pick_action_frame_in_shot(cap, start_f, end_f, max_checks=10):
    step = max(1, (end_f - start_f) // max_checks)
    best_score, best_frame, best_idx = -1, None, start_f

    for f_idx in range(start_f, end_f, step):
        f1 = frame_at(cap, f_idx)
        f2 = frame_at(cap, f_idx + step)
        if f1 is None or f2 is None:
            continue

        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))

        if mag > best_score:
            best_score = mag
            best_frame = f1
            best_idx = f_idx

    return best_idx, best_frame, best_score

# ----------------------
# KEYFRAME EXTRACTION
# ----------------------
def extract_keyframes_by_scenes(video_path, top_k_perc=0.5, max_keyframes=20):
    scenes = detect_scenes(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    collected = []

    for s_f, e_f in scenes:
        if e_f <= s_f:
            continue

        fidx, f, score = pick_action_frame_in_shot(cap, s_f, e_f, max_checks=40)
        if f is None:
            continue

        collected.append({
            "frame_no": fidx,
            "frame": cv2.cvtColor(f, cv2.COLOR_BGR2RGB),
            "score": score,
            "time_ms": (fidx / fps) * 1000
        })

    cap.release()

    # sort by motion score
    collected = sorted(collected, key=lambda x: x["score"], reverse=True)

    if max_keyframes:
        collected = collected[:max_keyframes]

    return collected

# ----------------------
# CLIP ENCODING
# ----------------------
def encode_frames_clip(frames_list):
    all_feats = []

    with torch.no_grad():
        for img in frames_list:
            img_tensor = clip_preprocess(Image.fromarray(img)).unsqueeze(0).to(device)
            feat = clip_model.encode_image(img_tensor)
            feat = F.normalize(feat, dim=-1)
            all_feats.append(feat.cpu())

    return torch.cat(all_feats, dim=0).to(device)

# ----------------------
# BLIP CAPTIONING
# ----------------------
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

# ----------------------
# FLAN-T5 SUMMARIZER (REPLACES FALCON)
# ----------------------
print("Loading FLAN-T5-Large summarizer...")
t5_name = "google/flan-t5-large"
t5_tokenizer = T5Tokenizer.from_pretrained(t5_name)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_name).to(device)

def summarize_with_t5(captions):
    text = "summarize: " + " ".join(captions)
    enc = t5_tokenizer(text, return_tensors="pt", truncation=True).to(device)

    out = t5_model.generate(
        enc.input_ids,
        max_new_tokens=200,
        num_beams=4,
        early_stopping=True
    )
    return t5_tokenizer.decode(out[0], skip_special_tokens=True)

# ----------------------
# MAIN PIPELINE
# ----------------------
video_path = "/content/video.mp4"

print("\nExtracting keyframes...")
collected = extract_keyframes_by_scenes(video_path, top_k_perc=0.5, max_keyframes=20)

frames = [c["frame"] for c in collected]
scores = np.array([c["score"] for c in collected])

print("Encoding frames with CLIP...")
embs = encode_frames_clip(frames)

sim = embs @ embs.T
uniqueness = 1 - sim.mean(dim=1).cpu().numpy()

uni_n = (uniqueness - uniqueness.min()) / (uniqueness.max() - uniqueness.min() + 1e-8)
mot_n = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

final_score = 0.6 * uni_n + 0.4 * mot_n
order = np.argsort(final_score)[::-1]

top_frames = [frames[i] for i in order]

# ----------------------
# CAPTIONING
# ----------------------
print("\nGenerating BLIP captions...")
captions = []
for f in top_frames:
    inp = processor(Image.fromarray(f), return_tensors="pt").to(device)
    out = blip.generate(**inp, max_new_tokens=50)
    captions.append(processor.decode(out[0], skip_special_tokens=True))

# ----------------------
# FLAN-T5 SUMMARY
# ----------------------
summary = summarize_with_t5(captions)

print("\n=============================")
print("FINAL SUMMARY")
print("=============================\n")
print(summary)
