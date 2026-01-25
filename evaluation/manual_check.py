import torch
import numpy as np
import os
from models.joint_model import JointEmbeddingModel

VIDEO_ID = "6dNh0Vp6JH8"
MODEL_PATH = "models/joint_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    model = JointEmbeddingModel().to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


model = load_model()

video_feat = torch.load(
    f"data/processed/video_features/{VIDEO_ID}.pt",
    map_location=DEVICE
).to(torch.float32)

scores = []


for f in os.listdir("data/processed/audio_features"):
    aid = f.replace(".npy", "")
    try:
        audio_feat = torch.tensor(
            np.load(f"data/processed/audio_features/{f}"),
            dtype=torch.float32
        ).to(DEVICE)
    except Exception as exc:
        print(f"Skipping {f}: {exc}")
        continue

    with torch.no_grad():
        v_emb, a_emb = model(
            video_feat.unsqueeze(0),
            audio_feat.unsqueeze(0)
        )
        v_emb = torch.nn.functional.normalize(v_emb, dim=1)
        a_emb = torch.nn.functional.normalize(a_emb, dim=1)
        sim = torch.sum(v_emb * a_emb).item()

    scores.append((aid, sim))

scores.sort(key=lambda x: x[1], reverse=True)
