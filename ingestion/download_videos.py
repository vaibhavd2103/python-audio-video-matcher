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
    url = f"https://www.youtube.com/shorts/{video_id}"
    output_path = os.path.join(OUTPUT_DIR, f"{video_id}.mp4")

    print(f"Downloading (audio+video only): {video_id}")

    cmd = [
        "yt-dlp",
        # 🔒 REQUIRE BOTH VIDEO + AUDIO
        "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]",
        "--merge-output-format", "mp4",

        # ❌ DO NOT FALL BACK TO VIDEO-ONLY
        "--no-playlist",
        "--no-part",

        # ⏱️ NEVER HANG
        "--socket-timeout", "10",
        "--retries", "3",
        "--fragment-retries", "3",
        "--skip-unavailable-fragments",
        "--abort-on-unavailable-fragment",

        # 📁 OUTPUT
        "--output", output_path,

        # 🧠 LOGGING
        "--quiet",
        "--no-warnings",

        url
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0 or not os.path.exists(output_path):
        print(f"❌ Skipped (no audio available): {video_id}")
        if os.path.exists(output_path):
            os.remove(output_path)
