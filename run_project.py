import os
import subprocess
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def run_command(command):
    """Run a shell command and print its output."""
    try:
        result = subprocess.run(command, shell=True, check=True, text=True)
        print(f"Command '{command}' executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing '{command}': {e}")
        exit(1)

def main():
    print("Starting the project pipeline...")

    # Step 1: Fetch metadata
    print("Fetching video metadata...")
    run_command("python -m ingestion.fetch_metadata")

    # Step 2: Download videos
    print("Downloading videos...")
    run_command("python -m ingestion.download_videos")

    # Step 3: Process videos
    print("Processing videos...")
    run_command("python -m ingestion.process_videos")

    # Step 4: Extract audio features
    print("Extracting audio features...")
    run_command("python -m ingestion.extract_audio_features")

    # Step 5: Extract video features
    print("Extracting video features...")
    run_command("python -m ingestion.extract_video_features")

    # Step 6: Train the model
    print("Training the model...")
    run_command("python -m training.train")

    print("Pipeline completed. Model is ready.")
    
    # Step 7: Optional - Serve the model via FastAPI
    print("Starting the API server...")
    run_command("uvicorn api.app:app --reload")
    
    print("API server is running.")

if __name__ == "__main__":
    main()