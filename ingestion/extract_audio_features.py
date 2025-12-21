import json
import os
from tqdm import tqdm
from utils.extract_audio_features import extract_audio_features

MANIFEST_PATH = "data/processed/manifest.json"
OUTPUT_PATH = "data/processed/audio_features.json"

with open(MANIFEST_PATH) as f:
    manifest = json.load(f)

results = []

for item in tqdm(manifest):
    audio_path = item["audio_path"]
    video_id = item["video_id"]

    try:
        features = extract_audio_features(audio_path)
        features["video_id"] = video_id
        results.append(features)
    except Exception as e:
        print(f"❌ Failed on {video_id}: {e}")

with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved audio features for {len(results)} videos")
