#attention_3d.py
import torch
import torch.nn as nn

class Keyed3DAttention(nn.Module):
    def __init__(self, key_dim=64):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 16, (1,5,5), padding=(0,2,2))
        self.conv2 = nn.Conv3d(16, 1, (1,5,5), padding=(0,2,2))
        self.relu = nn.ReLU()
        self.key_fc = nn.Linear(key_dim, 1)

    def forward(self, video, key):
        x = self.relu(self.conv1(video))
        x = self.relu(self.conv2(x))
         # 🔹 ADD THIS LINE
        x = torch.sigmoid(x)
        gate = torch.sigmoid(self.key_fc(key)).view(-1,1,1,1,1)
        return x * gate
