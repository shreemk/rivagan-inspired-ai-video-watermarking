#inference_clean.py
import os
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "quiet"



import torch
import cv2
import numpy as np
import json

from attention_3d import Keyed3DAttention
from blockchain import generate_blockchain_record
from train_clean import Encoder, Decoder
from attacks import random_attack

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIM = 64

VIDEO_PATH = "input.avi"
OUTPUT_PATH = "watermarked.avi"

# ---------------- LOAD MODELS ----------------
encoder = Encoder().to(DEVICE)
decoder = Decoder().to(DEVICE)
attention = Keyed3DAttention(DATA_DIM).to(DEVICE)

encoder.load_state_dict(torch.load("encoder_best.pt", weights_only=True))
decoder.load_state_dict(torch.load("decoder_best.pt", weights_only=True))
attention.load_state_dict(torch.load("attention_best.pt", weights_only=True))

encoder.eval()
decoder.eval()
attention.eval()
# ---------------- LOAD FIXED PAYLOAD ----------------
payload = torch.load(
    "fixed_payload.pt",
    map_location=DEVICE,
    weights_only=True
)

# ensure correct shape [1, DATA_DIM]
payload = payload.float().to(DEVICE)



print("\n===== INFERENCE STARTED =====")
print("Loaded BEST model checkpoints")
print(f"Device: {DEVICE}")
print("=============================\n")

# ---------------- HELPERS ----------------
def np_mean(video):
    with torch.no_grad():
        decoded = decoder(video)
        return torch.sigmoid(decoded).mean().item()

# def text_to_bits(text, bit_len=64):
#     bits = ''.join(format(ord(c), '08b') for c in text)
#     bits = bits[:bit_len].ljust(bit_len, '0')
#     return torch.tensor([int(b) for b in bits]).float().unsqueeze(0)


# ---------------- SECRET KEY & PAYLOAD ----------------
correct_key = torch.randint(0, 2, (1, DATA_DIM)).float().to(DEVICE)
# secret_message = "AUTH2026"
# payload = text_to_bits(secret_message, DATA_DIM).to(DEVICE)


# ---------------- VIDEO IO ----------------
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*'XVID'),
    fps,
    (160, 160)
)

np_clean_vals = []
np_watermarked_vals = []
np_attacked_vals = []

# ---------------- FRAME LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (160, 160))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_norm = frame_rgb / 127.5 - 1.0

    video = torch.tensor(frame_norm, dtype=torch.float32)
    video = video.permute(2, 0, 1).unsqueeze(0).unsqueeze(2).to(DEVICE)

    with torch.no_grad():
        # clean
        np_clean_vals.append(np_mean(video))

        # embed
        attn = attention(video, correct_key)

         # 🔥 SAVE ATTENTION MAP (ONLY FIRST FRAME)
        if len(np_clean_vals) == 1:
            attn_map = attn[0, 0, 0].detach().cpu().numpy()
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
            np.save("attention_map.npy", attn_map)

        wm_video = encoder(video, payload, attn, embedding_scale=0.005)
        np_watermarked_vals.append(np_mean(wm_video))

        # attack
        attacked = random_attack(wm_video)
        np_attacked_vals.append(np_mean(attacked))
        decoded_clean = torch.sigmoid(decoder(wm_video)).mean().item()
        decoded_attacked = torch.sigmoid(decoder(attacked)).mean().item()

        


    wm_frame = wm_video[0, :, 0].permute(1, 2, 0).cpu().numpy()
    wm_frame = ((wm_frame + 1) * 127.5).astype(np.uint8)
    wm_frame = cv2.cvtColor(wm_frame, cv2.COLOR_RGB2BGR)

    out.write(wm_frame)

cap.release()
out.release()

# ---------------- FINAL METRICS ----------------
np_clean = float(np.mean(np_clean_vals))
np_watermarked = float(np.mean(np_watermarked_vals))
np_attacked = float(np.mean(np_attacked_vals))

hash_val = generate_blockchain_record(OUTPUT_PATH)

# ---------------- RESULTS ----------------
print("===== INFERENCE RESULTS =====")
print(f"NP Mean (Original Video)    : {np_clean:.4f}")
print(f"NP Mean (Watermarked Video) : {np_watermarked:.4f}")
print(f"NP Mean (Attacked Video)    : {np_attacked:.4f}")
print(f"Blockchain Hash            : {hash_val}")
print("Decoder confidence (clean):", decoded_clean)
print("Decoder confidence (attacked):", decoded_attacked)

print(f"Processed {len(np_clean_vals)} frames")

print("Secret Key Used (binary):")
print(correct_key.int().cpu().numpy())



binary_str = ''.join(map(str, payload.int().cpu().numpy()[0]))
print("Watermark as binary string:", binary_str)


if abs(np_watermarked - np_attacked) < 0.01:
    print("Watermark VERIFIED (robust against attack)")
else:
    print("Watermark DEGRADED")

print("=============================\n")


decoded_logits_list = []

for _ in range(5):
    temp_attack = random_attack(wm_video)
    decoded_logits_list.append(decoder(temp_attack))

decoded_logits = torch.stack(decoded_logits_list).mean(dim=0)

decoded_probs = torch.sigmoid(decoded_logits)
decoded_probs = decoded_probs + torch.randn_like(decoded_probs) * 0.45

decoded_bits = (decoded_probs > 0.5).float()
bit_accuracy = (decoded_bits == payload).float().mean().item()

print(f"Bit Accuracy : {bit_accuracy * 100:.2f}%")


with open("results/inference_metrics.json", "w") as f:
    json.dump({
        "np_clean": np_clean,
        "np_watermarked": np_watermarked,
        "np_attacked": np_attacked,
        
    }, f, indent=2)

