import json
import os
import torch
from tqdm import tqdm
from utils.video_features import VideoEmbedder

MANIFEST_PATH = "data/processed/manifest.json"
OUT_DIR = "data/processed/video_features"

os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = VideoEmbedder(device=device)

with open(MANIFEST_PATH) as f:
    manifest = json.load(f)

for item in tqdm(manifest):
    video_id = item["video_id"]
    frame_dir = item["frames_dir"]

    out_path = os.path.join(OUT_DIR, f"{video_id}.pt")

    if os.path.exists(out_path):
        continue

    embedding = embedder.embed_frames(frame_dir)

    if embedding is None:
        continue

    torch.save(embedding.cpu(), out_path)

print("Video feature extraction completed.")
