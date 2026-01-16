import os
import json
import numpy as np
import torch
from torch.nn.functional import cosine_similarity

from models.joint_model import JointEmbeddingModel
from utils.video_features import extract_video_embedding

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "models/joint_model.pt"
AUDIO_FEATURE_DIR = "data/processed/audio_features"
MANIFEST_PATH = "data/processed/manifest.json"


def load_audio_embeddings():
    audio_embeddings = {}
    for file in os.listdir(AUDIO_FEATURE_DIR):
        if file.endswith(".npy"):
            video_id = file.replace(".npy", "")
            emb = np.load(os.path.join(AUDIO_FEATURE_DIR, file))
            audio_embeddings[video_id] = torch.tensor(emb, dtype=torch.float32)
    return audio_embeddings


def load_model():
    model = JointEmbeddingModel().to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def recommend_audio(video_path, top_k=5):
    model = load_model()
    audio_embeddings = load_audio_embeddings()

    # ---- extract video embedding ----
    video_emb = extract_video_embedding(video_path)
    video_emb = torch.tensor(video_emb, dtype=torch.float32).to(DEVICE)
    video_emb = torch.nn.functional.normalize(video_emb, dim=0)

    scores = []

    for vid, audio_emb in audio_embeddings.items():
        audio_emb = audio_emb.to(DEVICE)
        audio_emb = torch.nn.functional.normalize(audio_emb, dim=0)

        sim = cosine_similarity(video_emb.unsqueeze(0), audio_emb.unsqueeze(0))
        scores.append((vid, sim.item()))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


if __name__ == "__main__":
    TEST_VIDEO = "data/raw/videos/36e-flrsraU.mp4"

    results = recommend_audio(TEST_VIDEO)

    print("\n🎵 Recommended audio tracks:")
    for vid, score in results:
        print(f"{vid}  | similarity={score:.4f}")
