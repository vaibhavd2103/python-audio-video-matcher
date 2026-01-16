import os
import json
from tqdm import tqdm
from utils.frame_extractor import extract_frames
from utils.audio_extractor import extract_audio
import math

RAW_VIDEO_DIR = "data/raw/videos"
FRAME_DIR = "data/processed/frames"
AUDIO_DIR = "data/processed/audio"
MANIFEST_PATH = "data/processed/manifest.json"
SHORTS_METADATA_PATH = "data/raw/metadata/shorts_metadata.json"

manifest = []

os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

with open(SHORTS_METADATA_PATH) as f:
        shorts_data = json.load(f)

for video_file in tqdm(os.listdir(RAW_VIDEO_DIR)):
    if not video_file.endswith(".mp4"):
        continue

    video_id = os.path.splitext(video_file)[0]
    video_path = os.path.join(RAW_VIDEO_DIR, video_file)

    frame_out = os.path.join(FRAME_DIR, video_id)
    audio_out = os.path.join(AUDIO_DIR, f"{video_id}.wav")

    frames_extracted = extract_frames(video_path, frame_out)
    has_audio = extract_audio(video_path, audio_out)
    
    if frames_extracted == 0:
        continue

    if not has_audio or not os.path.exists(audio_out):
        continue  # HARD FILTER — correct behavior

    # ---------- Engagement ----------

    video_meta = next((item for item in shorts_data if item["video_id"] == video_id), None)
    if not video_meta:
        continue  # SKIP — no metadata found

    views = video_meta.get("views", 0)
    likes = video_meta.get("likes", 0)
    comments = video_meta.get("comments", 0)

    engagement = (
        0.4 * math.log(views + 1)
        + 0.4 * (likes / (views + 1))
        + 0.2 * (comments / (views + 1))
    )
    
    engagement = max(0.0, min(engagement, 10.0))
    
    manifest.append({
        "video_id": video_id,
        "video_path": video_path,
        "frames_dir": frame_out,
        "audio_path": audio_out,
        "num_frames": frames_extracted,
        "engagement": engagement
    })

with open(MANIFEST_PATH, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Processed {len(manifest)} valid videos")
