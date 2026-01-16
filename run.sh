python -m ingestion.fetch_metadata
python -m ingestion.download_videos
python -m ingestion.process_videos
python -m ingestion.extract_audio_features
python -m ingestion.extract_video_features
python -m training.train