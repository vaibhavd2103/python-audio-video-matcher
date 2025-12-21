import json
import os
from tqdm import tqdm
from utils.extract_audio_features import extract_audio_features

MANIFEST_PATH = "data/processed/manifest.json"
FEATURE_DIR = "data/processed/audio_features"

os.makedirs(FEATURE_DIR, exist_ok=True)

with open(MANIFEST_PATH) as f:
    manifest = json.load(f)

for item in tqdm(manifest):
    audio_path = item["audio_path"]
    video_id = item["video_id"]

    feature_path = os.path.join(FEATURE_DIR, f"{video_id}.npy")

    if not os.path.exists(audio_path):
        continue

    features = extract_audio_features(audio_path)
    features.dump(feature_path)
