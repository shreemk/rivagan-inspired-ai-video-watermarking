# plot_results.py
import json
import matplotlib.pyplot as plt

with open("results/metrics.json") as f:
    data = json.load(f)

epochs = [d["epoch"] for d in data]
train_loss = [d["train_loss"] for d in data]
val_loss = [d["val_loss"] for d in data]
psnr_vals = [d["psnr"] for d in data]
ssim_vals = [d["ssim"] for d in data]

# -------- LOSS PLOT --------
plt.figure(figsize=(7,5))
plt.plot(epochs, train_loss, marker='o', label="Training Loss")
plt.plot(epochs, val_loss, marker='s', label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/train_val_loss.png", dpi=300)
plt.show()

# -------- QUALITY PLOT --------
plt.figure(figsize=(7,5))
plt.plot(epochs, psnr_vals, marker='o', label="PSNR (dB)")
plt.plot(epochs, ssim_vals, marker='s', label="SSIM")
plt.xlabel("Epoch")
plt.ylabel("Perceptual Quality")
plt.title("Perceptual Quality vs Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/psnr_ssim.png", dpi=300)
plt.show()
