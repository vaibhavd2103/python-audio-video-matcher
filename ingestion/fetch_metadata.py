import os
import json
from googleapiclient.discovery import build
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=API_KEY)

OUTPUT_FILE = "data/raw/metadata/shorts_metadata.json"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

def fetch_shorts(query="songs", max_results=10):
    request = youtube.search().list(
        part="snippet",
        q=query,
        type="video",
        videoDuration="short",
        maxResults=max_results,
    )
    response = request.execute()
    video_ids = [item["id"]["videoId"] for item in response["items"]]
    
    print("Response from the API:", response)

    stats = youtube.videos().list(
        part="statistics,contentDetails",
        id=",".join(video_ids)
    ).execute()
    print("Statistics response from the API:", stats)

    videos = []
    for item in stats["items"]:
        stats = item["statistics"]
        videos.append({
            "video_id": item["id"],
            "views": int(stats.get("viewCount", 0)),
            "likes": int(stats.get("likeCount", 0)),
            "comments": int(stats.get("commentCount", 0)),
        })
        print(f"Fetched metadata for video ID: {item['id']}")

    return videos

if __name__ == "__main__":
    data = fetch_shorts(max_results=10)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(data)} video metadata entries")
