import os
import torch
import numpy as np
from tqdm import tqdm

from models.joint_model import JointEmbeddingModel

VIDEO_DIR = "data/processed/video_features"
AUDIO_DIR = "data/processed/audio_features"
MODEL_PATH = "models/joint_model.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

video_ids = [
    f.replace(".pt", "")
    for f in os.listdir(VIDEO_DIR)
    if f.endswith(".pt")
]

def cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=1)


def load_model():
    model = JointEmbeddingModel().to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

ranks = []

model = load_model()

audio_features = {}
for f in os.listdir(AUDIO_DIR):
    if f.endswith(".npy"):
        aid = f.replace(".npy", "")
        audio_features[aid] = torch.tensor(
            np.load(os.path.join(AUDIO_DIR, f)),
            dtype=torch.float32
        ).to(DEVICE)

for vid in tqdm(video_ids):
    video_feat = torch.load(
        os.path.join(VIDEO_DIR, f"{vid}.pt"),
        map_location=DEVICE
    ).to(torch.float32)

    sims = []

    with torch.no_grad():
        for aid, audio_feat in audio_features.items():
            v_emb, a_emb = model(
                video_feat.unsqueeze(0),
                audio_feat.unsqueeze(0)
            )
            v_emb = torch.nn.functional.normalize(v_emb, dim=1)
            a_emb = torch.nn.functional.normalize(a_emb, dim=1)
            sims.append((aid, cosine_sim(v_emb, a_emb).item()))

    sims.sort(key=lambda x: x[1], reverse=True)

    ranked_ids = [x[0] for x in sims]
    rank = ranked_ids.index(vid) + 1
    ranks.append(rank)

ranks = np.array(ranks)

print(f"Recall@1  : {(ranks <= 1).mean():.3f}")
print(f"Recall@5  : {(ranks <= 5).mean():.3f}")
print(f"Recall@10 : {(ranks <= 10).mean():.3f}")
print(f"Mean Rank : {ranks.mean():.2f}")
