import os
import json
import re
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=API_KEY)

OUTPUT_FILE = "data/raw/metadata/shorts_metadata.json"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

def parse_duration(duration):
    match = re.match(r"PT(?:(\d+)M)?(?:(\d+)S)?", duration)
    if not match:
        return 0
    minutes = int(match.group(1) or 0)
    seconds = int(match.group(2) or 0)
    return minutes * 60 + seconds

def fetch_shorts_with_audio_bias(query="songs shorts", max_results=300):
    # The Search API caps maxResults to 50 per request; paginate to reach max_results.
    video_ids = []
    next_page_token = None

    while len(video_ids) < max_results:
        batch_size = min(50, max_results - len(video_ids))
        request = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            videoDuration="short",
            maxResults=batch_size,
            pageToken=next_page_token,
            # order="viewCount"
        )
        response = request.execute()
        video_ids.extend(item["id"]["videoId"] for item in response.get("items", []))
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    results = []

    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i + 50]
        details = youtube.videos().list(
            part="statistics,contentDetails,status",
            id=",".join(chunk)
        ).execute()

        for item in details.get("items", []):
            status = item["status"]
            content = item["contentDetails"]

            # Exclusions
            if status.get("uploadStatus") != "processed":
                continue

            if status.get("privacyStatus") != "public":
                continue

            duration = parse_duration(content.get("duration", "PT0S"))
            if duration == 0 or duration > 60:
                continue  # Exclude videos longer than 1 minute

            # if not content.get("hasAudio", True):
            #     continue  # Exclude videos without audio

            stats = item.get("statistics", {})

            results.append({
                "video_id": item["id"],
                "views": int(stats.get("viewCount", 0)),
                "likes": int(stats.get("likeCount", 0)),
                "comments": int(stats.get("commentCount", 0)),
            })

    return results

if __name__ == "__main__":
    data = fetch_shorts_with_audio_bias()
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(data)} audio-biased video entries")