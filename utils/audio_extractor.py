import os
import subprocess

def extract_audio(video_path: str, output_audio_path: str) -> bool:
    """
    Extract audio using ffmpeg.
    Returns True if audio exists and extraction succeeds.
    """

    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "22050",
        "-f", "wav",
        output_audio_path
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 0
