# attacks.py
import torch
import torch.nn.functional as F
import random

def random_attack(video):
    """
    Strong attack pipeline
    video shape: [B, C, D, H, W]
    Temporal dimension (D) is preserved
    """

    B, C, D, H, W = video.shape

    # 1️⃣ Random spatial resize (downscale + upscale)
    if random.random() < 0.5:
        scale = random.uniform(0.7, 0.9)
        new_h, new_w = int(H * scale), int(W * scale)
        video = F.interpolate(video, size=(D, new_h, new_w),
                               mode="trilinear", align_corners=False)
        video = F.interpolate(video, size=(D, H, W),
                               mode="trilinear", align_corners=False)

    # 2️⃣ Random crop + pad back
    if random.random() < 0.4:
        crop = random.randint(5, 15)
        video = video[:, :, :, crop:H-crop, crop:W-crop]
        video = F.pad(video, (crop, crop, crop, crop))

    # 3️⃣ Gaussian noise
    if random.random() < 0.5:
        noise = torch.randn_like(video) * 0.05
        video = video + noise

    # 4️⃣ Blur (average pooling)
    if random.random() < 0.3:
        video = F.avg_pool3d(video, kernel_size=(1,3,3), stride=1, padding=(0,1,1))

    # 5️⃣ Quantization (JPEG-like compression)
    if random.random() < 0.4:
        video = torch.round(video * 16) / 16

    return video
