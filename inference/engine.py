import os
import numpy as np
import torch

from models.joint_model import JointEmbeddingModel
from utils.video_features import extract_video_embedding

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "models/joint_model.pt"
AUDIO_FEATURE_DIR = "data/processed/audio_features"

# ---------- load once ----------
_model = None
_audio_features = None


def load_engine():
    global _model, _audio_features

    if _model is None:
        _model = JointEmbeddingModel().to(DEVICE)
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        _model.load_state_dict(ckpt["model_state"])
        _model.eval()

    if _audio_features is None:
        _audio_features = {}
        for f in os.listdir(AUDIO_FEATURE_DIR):
            if f.endswith(".npy"):
                vid = f.replace(".npy", "")
                _audio_features[vid] = torch.tensor(
                    np.load(os.path.join(AUDIO_FEATURE_DIR, f)),
                    dtype=torch.float32
                ).to(DEVICE)

    return _model, _audio_features


def recommend_audio(video_path: str, top_k: int = 5):
    model, audio_features = load_engine()

    # ---- video → embedding via model ----
    video_feat = extract_video_embedding(video_path)
    video_tensor = torch.tensor(video_feat).unsqueeze(0).to(DEVICE)

    results = []

    with torch.no_grad():
        for audio_id, audio_tensor in audio_features.items():
            v_emb, a_emb = model(video_tensor, audio_tensor.unsqueeze(0))

            v_emb = torch.nn.functional.normalize(v_emb, dim=1)
            a_emb = torch.nn.functional.normalize(a_emb, dim=1)

            score = torch.sum(v_emb * a_emb).item()
            results.append((audio_id, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]
