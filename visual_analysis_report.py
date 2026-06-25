import os
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "quiet"

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import json

from train_clean import Encoder, Decoder
from attention_3d import Keyed3DAttention

# ---------------- SETTINGS ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIM = 64

VIDEO_PATH = "watermarked.avi"
MODEL_DIR = "models"
RESULT_DIR = "results"
METRICS_PATH = "results/metrics.json"

os.makedirs(RESULT_DIR, exist_ok=True)

# ---------------- LOAD MODELS ----------------
encoder = Encoder().to(DEVICE)
decoder = Decoder().to(DEVICE)
attention = Keyed3DAttention(DATA_DIM).to(DEVICE)

encoder.load_state_dict(torch.load(f"{MODEL_DIR}/encoder.pt", weights_only=True))
decoder.load_state_dict(torch.load(f"{MODEL_DIR}/decoder.pt", weights_only=True))
attention.load_state_dict(torch.load(f"{MODEL_DIR}/attention.pt", weights_only=True))

encoder.eval()
decoder.eval()
attention.eval()

# ---------------- LOAD ONE FRAME ----------------
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Failed to read video frame")

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_resized = cv2.resize(frame_rgb, (160, 160))

video = torch.tensor(frame_resized / 127.5 - 1.0, dtype=torch.float32)
video = video.permute(2, 0, 1).unsqueeze(0).unsqueeze(2).to(DEVICE)

# ---------------- FIXED KEY (for visualization only) ----------------
torch.manual_seed(42)
secret_key = torch.randint(0, 2, (1, DATA_DIM)).float().to(DEVICE)

# ---------------- ATTENTION MAP ----------------
with torch.no_grad():
    attn = attention(video, secret_key)

attn_map = attn[0, 0, 0].detach().cpu().numpy()
attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

# ---------------- DECODER CONFIDENCE ----------------
with torch.no_grad():
    decoded_logits = decoder(video)
    decoded_probs = torch.sigmoid(decoded_logits)
    np_mean = decoded_probs.mean().item()

print("\nVISUAL ANALYSIS METRICS")
print(f"NP Mean (Decoder Confidence): {np_mean:.4f}")

# ---------------- VISUAL 1: ATTENTION HEATMAP ----------------
plt.figure(figsize=(6, 6))
plt.imshow(frame_rgb)
plt.imshow(
    cv2.resize(attn_map, (frame_rgb.shape[1], frame_rgb.shape[0])),
    cmap="jet",
    alpha=0.55
)
plt.title("Attention-Guided Watermark Embedding Regions")
plt.axis("off")
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/attention_heatmap.png", dpi=300)
plt.close()

# ---------------- LOAD TRAINING METRICS ----------------
with open(METRICS_PATH) as f:
    history = json.load(f)

epochs = [h["epoch"] for h in history]
train_loss = [h["train_loss"] for h in history]
val_loss = [h["val_loss"] for h in history]
psnr_vals = [h["psnr"] for h in history]
ssim_vals = [h["ssim"] for h in history]

# ---------------- VISUAL 2: LOSS CURVES ----------------
plt.figure(figsize=(7, 5))
plt.plot(epochs, train_loss, label="Training Loss")
plt.plot(epochs, val_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/loss_curves.png", dpi=300)
plt.close()

# ---------------- VISUAL 3: QUALITY METRICS ----------------
plt.figure(figsize=(7, 5))
plt.plot(epochs, psnr_vals, label="PSNR (dB)")
plt.plot(epochs, ssim_vals, label="SSIM")
plt.xlabel("Epoch")
plt.ylabel("Quality")
plt.title("Perceptual Quality vs Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/quality_metrics.png", dpi=300)
plt.close()

print("\nVISUAL ANALYSIS COMPLETE.")
print("Saved:")
print(" - attention_heatmap.png")
print(" - loss_curves.png")
print(" - quality_metrics.png")
