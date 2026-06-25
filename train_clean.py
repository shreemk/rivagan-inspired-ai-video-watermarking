#train_clean.py
import os
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "quiet"

import torch, os
import numpy as np
import json
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from attention_3d import Keyed3DAttention
from attacks import random_attack
from dataset import VideoDataset   # reuse your existing dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIM = 64
EPOCHS = 20
LR = 0.0005
no_improve_epoch = 0

def compute_quality_metrics(original, watermarked):
    """
    original, watermarked: tensors of shape [B, C, D, H, W]
    Returns: PSNR, SSIM
    """

    # Take first sample and first temporal frame
    orig = original[0, :, 0].permute(1, 2, 0).detach().cpu().numpy()
    wm   = watermarked[0, :, 0].permute(1, 2, 0).detach().cpu().numpy()

    # Convert from [-1, 1] to [0, 1]
    orig = (orig + 1.0) / 2.0
    wm   = (wm + 1.0) / 2.0

    # Safety clip
    orig = orig.clip(0, 1)
    wm   = wm.clip(0, 1)

    psnr_val = psnr(orig, wm, data_range=1.0)
    ssim_val = ssim(orig, wm, channel_axis=2, data_range=1.0)

    return psnr_val, ssim_val

# ---------------- MODELS ----------------
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(4, 3, 3, padding=1)

    def forward(self, video, data, attn, embedding_scale):
        # RGB → Grayscale
        gray = 0.299 * video[:, 0:1] + 0.587 * video[:, 1:2] + 0.114 * video[:, 2:3]

        # Payload → spatial map
        # Expand each bit spatially
        payload_map = data.view(-1, DATA_DIM, 1, 1, 1)
        payload_map = payload_map.mean(dim=1, keepdim=True)  # optional smoothing
        payload_map = payload_map * attn


        # Concatenate original video + payload
        combined = torch.cat([video, payload_map], dim=1)

        residual = self.conv(combined)
        return video + embedding_scale * residual



class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv3d(16, DATA_DIM, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, video):
        """
        video: [B, 3, T, H, W]
        Output: [B, DATA_DIM] watermark bits
        """
        x = self.relu(self.conv1(video))
        x = self.conv2(x)

        # 🔹 DUPLICATION-BASED DECODING
        # aggregate over space & time (inverse of duplication)
        x = x.mean(dim=[2, 3, 4])   # [B, DATA_DIM]

        return x


class Adversary(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(3, 1, 3, padding=1)

    def forward(self, x):
        return torch.sigmoid(self.conv(x)).mean()

# ---------------- INIT ----------------
encoder = Encoder().to(DEVICE)
decoder = Decoder().to(DEVICE)
adversary = Adversary().to(DEVICE)
attention = Keyed3DAttention(DATA_DIM).to(DEVICE)

opt = torch.optim.Adam(
    list(encoder.parameters()) +
    list(decoder.parameters()) +
    list(attention.parameters()),
    lr=LR
)

if __name__ == "__main__":
    # training code here
    train_dataset = VideoDataset("C:/Users/Nithu/Pictures/hollywood2/train")
    val_dataset = VideoDataset("C:/Users/Nithu/Pictures/hollywood2/val")

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    fixed_payload = torch.randint(0, 2, (1, DATA_DIM)).float().to(DEVICE)
    torch.save(fixed_payload, "fixed_payload.pt")



# ---------------- TRAINING CONFIG PRINT (👉 HERE) ----------------
    print("\n========== TRAINING CONFIGURATION ==========")
    print("Objective      : Invisible & Robust Video Watermarking")
    print("Security       : Secret key + Blockchain-ready")
    print(f"Dataset (Train): C:/Users/Nithu/Pictures/hollywood2/train")
    print(f"Dataset (Val)  : C:/Users/Nithu/Pictures/hollywood2/val")
    print(f"Epochs         : {EPOCHS}")
    print(f"Learning Rate  : {LR}")
    print(f"Batch Size     : {train_loader.batch_size}")
    print(f"Device         : {DEVICE}")
    print("===========================================\n")

# ---------------- TRAIN ----------------
    history = []
    best_psnr = -1
    
    for epoch in range(EPOCHS):
        psnr_vals = []
        ssim_vals = []
        train_losses = []

        #-----set the embedding strength----
        if epoch < 5:
            embedding_scale = 0.02 
        else:
            embedding_scale = 0.008 
    
        #-----train mode---
        encoder.train()
        decoder.train()
        attention.train()

        #--------batch loop----
        for video in train_loader:
            video = video.to(DEVICE)
            payload = fixed_payload.repeat(video.size(0),1)
            noise_mask = (torch.rand_like(payload)<0.1).float()
            data = torch.abs(payload - noise_mask)
            key  = torch.randint(0, 2, (video.size(0), DATA_DIM)).float().to(DEVICE)

        # 1️⃣ Attention
            attn = attention(video, key)

        # 2️⃣ Watermark embedding
            wm = encoder(video, data, attn, embedding_scale)
           

        # 3️⃣ Quality metrics (on clean watermark)
            psnr_val, ssim_val = compute_quality_metrics(video, wm)
            psnr_vals.append(psnr_val)
            ssim_vals.append(ssim_val)

        # 4️⃣ Simulate attack
            
            wm_attacked = random_attack(wm)


        # 5️⃣ Decode watermark
            decoded = decoder(wm_attacked)

        # 6️⃣ Base decoding loss
            bce = torch.nn.BCEWithLogitsLoss()(decoded, data)
            mse = torch.mean((torch.sigmoid(decoded) - data) ** 2)
            loss = 2.0 * bce + 0.1 * mse
            loss = loss + 0.001 * torch.mean(torch.sigmoid(decoded)**2)



        # 7️⃣ Quality-aware penalty (ADD AFTER loss exists)
            quality_penalty = max(0, 25 - psnr_val) * 0.001
            loss = loss + quality_penalty

        # 8️⃣ Backpropagation
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_losses.append(loss.item())

        # -------- VALIDATION --------
        encoder.eval()
        decoder.eval()
        attention.eval()

        val_losses = []

        with torch.no_grad():
            for video in val_loader:
                video = video.to(DEVICE)
                data = fixed_payload.repeat(video.size(0), 1)
                key  = torch.randint(0,2,(video.size(0),DATA_DIM)).float().to(DEVICE)

                attn = attention(video, key)
                wm = encoder(video, data, attn, embedding_scale)

                
                wm_attacked = random_attack(wm)
                
                decoded = decoder(wm_attacked)


                loss = torch.nn.BCEWithLogitsLoss()(decoded, data)
                val_losses.append(loss.item())

        avg_psnr = float(np.mean(psnr_vals))
        avg_ssim = float(np.mean(ssim_vals))


        history.append({
            "epoch": epoch + 1,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": float(np.mean(val_losses)),
            "psnr": float(np.mean(psnr_vals)),
            "ssim": float(np.mean(ssim_vals))
            })

        # SAVE BEST MODEL (ADD HERE ✅)
        if epoch == 0 or avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(encoder.state_dict(), "encoder_best.pt")
            torch.save(decoder.state_dict(), "decoder_best.pt")
            torch.save(attention.state_dict(), "attention_best.pt")

        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {np.mean(train_losses):.4f} | "
            f"Val Loss: {np.mean(val_losses):.4f} | "
            f"PSNR: {np.mean(psnr_vals):.2f} dB | "
            f"SSIM: {np.mean(ssim_vals):.4f}"
            )

    print("\n===== TRAINING SUMMARY =====")
    print(f"Best PSNR achieved      : {best_psnr:.2f} dB")
    print("Model selection based on perceptual quality (PSNR)")
    print("Observed trade-off between invisibility and robustness")
    print("Best model checkpoints saved for inference")
    print("============================\n")

    # ---------------- SAVE ----------------
    os.makedirs("models", exist_ok=True)
    torch.save(encoder.state_dict(), "models/encoder.pt")
    torch.save(decoder.state_dict(), "models/decoder.pt")
    torch.save(attention.state_dict(), "models/attention.pt")

    os.makedirs("results", exist_ok=True)

    with open("results/metrics.json", "w") as f:
        json.dump(history, f, indent=2)

