# dataset.py
import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class VideoDataset(Dataset):
    def __init__(self, root_dir, size=(160,160)):
        self.root_dir = root_dir
        self.files = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".avi")
        ]
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.files[idx])
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_id = np.random.randint(0, max(1, length))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()

        cap.release()

        if not ret:
            raise RuntimeError(f"Failed to read {self.files[idx]}")

        frame = cv2.resize(frame, self.size)
        frame = frame.astype(np.float32) / 127.5 - 1.0

        # shape: [3, T=1, H, W]
        frame = torch.from_numpy(frame).permute(2,0,1).unsqueeze(1)
        return frame
