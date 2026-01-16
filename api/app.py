import os
import torch
import shutil
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from models.joint_model import JointEmbeddingModel
from fastapi.responses import FileResponse

from inference.engine import recommend_audio

AUDIO_DIR = "data/processed/audio"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "models/joint_model.pt"
AUDIO_FEATURE_DIR = "data/processed/audio_features"
UPLOAD_DIR = "api/uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Audio Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Load model once --------
model = JointEmbeddingModel().to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

# -------- Load audio features once --------
audio_embeddings = {}
for f in os.listdir(AUDIO_FEATURE_DIR):
    if f.endswith(".npy"):
        vid = f.replace(".npy", "")
        audio_embeddings[vid] = torch.tensor(
            np.load(os.path.join(AUDIO_FEATURE_DIR, f)),
            dtype=torch.float32
        ).to(DEVICE)


@app.post("/recommend")
async def recommend(file: UploadFile = File(...), top_k: int = 5):
    video_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    results = recommend_audio(video_path, top_k)

    return [
    {
        "audio_id": aid,
        "score": score,
        "audio_url": f"http://127.0.0.1:8000/audio/{aid}"
    }
    for aid, score in results
]

@app.get("/audio/{audio_id}")
def get_audio(audio_id: str):
    audio_path = os.path.join(AUDIO_DIR, f"{audio_id}.wav")

    if not os.path.exists(audio_path):
        return {"error": "Audio not found"}

    return FileResponse(audio_path, media_type="audio/wav")
