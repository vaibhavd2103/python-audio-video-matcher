import torch
import numpy as np
import os
from models.joint_model import JointEmbeddingModel

VIDEO_ID = "8r6KiemOdRM"
MODEL_PATH = "models/joint_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    model = JointEmbeddingModel().to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


model = load_model()

video_path = f"data/processed/video_features/{VIDEO_ID}.pt"
audio_path = f"data/processed/audio_features/{VIDEO_ID}.npy"

if not os.path.exists(video_path):
    raise FileNotFoundError(f"Missing video features: {video_path}")

video_feat = torch.load(
    video_path,
    map_location=DEVICE
).to(torch.float32)

scores = []

with torch.no_grad():
    v_emb = model.video_proj(video_feat.unsqueeze(0))
    v_emb = torch.nn.functional.normalize(v_emb, dim=1)

for f in os.listdir("data/processed/audio_features"):
    if not f.endswith(".npy"):
        continue
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
        a_emb = model.audio_proj(audio_feat.unsqueeze(0))
        a_emb = torch.nn.functional.normalize(a_emb, dim=1)
        sim = torch.sum(v_emb * a_emb).item()

    scores.append((aid, sim))

scores.sort(key=lambda x: x[1], reverse=True)

print(f"Top audio matches for video {VIDEO_ID}:")
for aid, score in scores[:5]:
    print(aid, f"{score:.3f}")

if os.path.exists(audio_path):
    ranked_ids = [x[0] for x in scores]
    true_rank = ranked_ids.index(VIDEO_ID) + 1
    true_score = next(score for aid, score in scores if aid == VIDEO_ID)
    print(f"\nExact audio ID rank: {true_rank} | score={true_score:.3f}")
else:
    print(f"\nExact audio ID not found in audio features: {audio_path}")
