import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset

class AVEmbeddingDataset(Dataset):
    def __init__(self, manifest_path):
        with open(manifest_path) as f:
            self.data = json.load(f)

        self.video_dir = "data/processed/video_features"
        self.audio_dir = "data/processed/audio_features"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        vid = item["video_id"]

        video_emb = torch.load(
            os.path.join(self.video_dir, f"{vid}.pt")
        )

        audio_emb = torch.tensor(
            np.load(os.path.join(self.audio_dir, f"{vid}.npy"))
        )

        engagement = torch.tensor(
            item["engagement"], dtype=torch.float32
        )

        return video_emb, audio_emb, engagement
