import json
import os
import subprocess
from tqdm import tqdm

METADATA_FILE = "data/raw/metadata/shorts_metadata.json"
OUTPUT_DIR = "data/raw/videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(METADATA_FILE) as f:
    videos = json.load(f)

for video in tqdm(videos):
    video_id = video["video_id"]
    url = f"https://www.youtube.com/watch?v={video_id}"

    cmd = [
        "yt-dlp",
        "--merge-output-format", "mp4",
        "--output", f"{OUTPUT_DIR}/{video_id}.mp4",
        "--match-filter", "duration < 60",
        url
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
