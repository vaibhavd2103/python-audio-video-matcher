import os
import torch
import numpy as np
from tqdm import tqdm

VIDEO_DIR = "data/processed/video_features"
AUDIO_DIR = "data/processed/audio_features"

video_ids = [
    f.replace(".pt", "")
    for f in os.listdir(VIDEO_DIR)
    if f.endswith(".pt")
]

def cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0)

ranks = []

for vid in tqdm(video_ids):
    video_emb = torch.load(os.path.join(VIDEO_DIR, f"{vid}.pt"))

    sims = []

    for aid in video_ids:
        audio_emb = torch.tensor(
            np.load(os.path.join(AUDIO_DIR, f"{aid}.npy"))
        )
        sims.append((aid, cosine_sim(video_emb, audio_emb).item()))

    sims.sort(key=lambda x: x[1], reverse=True)

    ranked_ids = [x[0] for x in sims]
    rank = ranked_ids.index(vid) + 1
    ranks.append(rank)

ranks = np.array(ranks)

print(f"Recall@1  : {(ranks <= 1).mean():.3f}")
print(f"Recall@5  : {(ranks <= 5).mean():.3f}")
print(f"Recall@10 : {(ranks <= 10).mean():.3f}")
print(f"Mean Rank : {ranks.mean():.2f}")
