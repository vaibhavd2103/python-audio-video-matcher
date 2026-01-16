import torch
import numpy as np
import os

VIDEO_ID = "your_video_id_here"

video_emb = torch.load(f"data/processed/video_features/{VIDEO_ID}.pt")

scores = []

for f in os.listdir("data/processed/audio_features"):
    aid = f.replace(".npy", "")
    audio_emb = torch.tensor(
        np.load(f"data/processed/audio_features/{f}")
    )

    sim = torch.nn.functional.cosine_similarity(video_emb, audio_emb, dim=0)
    scores.append((aid, sim.item()))

scores.sort(key=lambda x: x[1], reverse=True)

print("Top 5 recommended audios:")
for aid, score in scores[:5]:
    print(aid, f"{score:.3f}")
