import os
import json
from tqdm import tqdm
from utils.frame_extractor import extract_frames
from utils.audio_extractor import extract_audio

RAW_VIDEO_DIR = "data/raw/videos"
FRAME_DIR = "data/processed/frames"
AUDIO_DIR = "data/processed/audio"
MANIFEST_PATH = "data/processed/manifest.json"

manifest = []

os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

for video_file in tqdm(os.listdir(RAW_VIDEO_DIR)):
    if not video_file.endswith(".mp4"):
        continue

    video_id = os.path.splitext(video_file)[0]
    video_path = os.path.join(RAW_VIDEO_DIR, video_file)

    frame_out = os.path.join(FRAME_DIR, video_id)
    audio_out = os.path.join(AUDIO_DIR, f"{video_id}.wav")

    frames_extracted = extract_frames(video_path, frame_out)
    has_audio = extract_audio(video_path, audio_out)

    if not has_audio:
        continue  # HARD FILTER — correct behavior

    manifest.append({
        "video_id": video_id,
        "video_path": video_path,
        "frames_dir": frame_out,
        "audio_path": audio_out,
        "num_frames": frames_extracted
    })

with open(MANIFEST_PATH, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Processed {len(manifest)} valid videos")
