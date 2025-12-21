import json
import os
import numpy as np
from tqdm import tqdm
from utils.audio_features import extract_audio_features

MANIFEST_PATH = "data/processed/manifest.json"
OUT_DIR = "data/processed/audio_features"

os.makedirs(OUT_DIR, exist_ok=True)

with open(MANIFEST_PATH) as f:
    manifest = json.load(f)

success = 0

for item in tqdm(manifest):
    video_id = item["video_id"]
    audio_path = item["audio_path"]

    out_path = os.path.join(OUT_DIR, f"{video_id}.npy")

    if not os.path.exists(audio_path):
        continue

    if os.path.exists(out_path):
        continue

    try:
        features = extract_audio_features(audio_path)
        np.save(out_path, features)
        success += 1
    except Exception as e:
        print(f"❌ Failed on {video_id}: {e}")

print(f"✅ Saved audio features for {success} videos")
