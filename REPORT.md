# Python Audio-Video Matcher: Technical Report

## Abstract
This report describes a Python-based system that ingests short-form videos, extracts audio and visual features, learns a joint embedding space, and serves audio recommendations for an uploaded video. The pipeline integrates YouTube metadata collection, media download, preprocessing, feature extraction, model training, evaluation, and a FastAPI + Streamlit interface. The system aligns audio and video embeddings using an engagement-weighted contrastive objective and returns the top-k most similar audio tracks for a query video. References and concrete examples are provided for each major component.

## Introduction
Short-form video platforms rely on matching audio to visuals for recommendation, reuse, and trend analysis. This project implements a full-stack prototype that builds a paired audio-video embedding space from Shorts-style content and exposes recommendations through an API and a lightweight web UI. The pipeline emphasizes reproducibility and modularity with separate ingestion, processing, training, evaluation, and inference modules. The system is orchestrated by a single runner script. [R1]

Example (end-to-end run):
```bash
python run_project.py
```

## Problem Statement
Given a collection of short videos with audio, build a system that can recommend the most relevant audio tracks for a new video clip. The system must (1) ingest and filter short videos that contain audio, (2) compute consistent audio and visual embeddings, (3) learn a joint embedding space where matching audio-video pairs are close, and (4) serve top-k audio recommendations for a new video. [R1][R2][R3][R4]

## Objective
- Collect metadata for short videos and filter for valid, public, short-duration items. [R2]
- Download short videos with both audio and video tracks. [R3]
- Extract frames and audio, then compute fixed-length audio and video embeddings. [R4][R5][R6][R7]
- Train a joint embedding model with an engagement-aware contrastive loss. [R8][R9]
- Provide an API and UI for inference and audio playback. [R10][R11][R12]

## Background and Related Work
This project uses common audio-visual representation learning ideas: CNN-based visual encoders (e.g., ResNet features) and MFCC-based audio embeddings, combined in a shared space via contrastive learning. It also uses standard retrieval evaluation (recall@k, mean rank) for audio-video matching and a typical API + UI stack for deployment.

Examples in this codebase:
- Visual features via ResNet18 frame embeddings with temporal mean pooling. [R7]
- Audio features via MFCCs with mean pooling. [R6]
- Retrieval evaluation using cosine similarity and recall@k. [R13]

## Methodology

### 1) Data Collection and Ingestion
- Metadata is fetched from the YouTube Data API using a query such as "songs shorts" and restricted to short-duration videos (<= 60 seconds). [R2]
- Metadata is stored as JSON with view/like/comment statistics. [R2]
- Videos are downloaded using `yt-dlp` with audio+video format constraints to avoid silent videos. [R3]

Example metadata filter logic:
```python
if duration == 0 or duration > 60:
    continue  # Exclude videos longer than 1 minute
```
[R2]

Example download command pattern:
```bash
yt-dlp -f "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]" --merge-output-format mp4 ...
```
[R3]

### 2) Processing and Feature Extraction
- Frames are sampled from each video at a fixed FPS with a max frame cap; audio is extracted to a mono 22.05 kHz WAV file. [R4][R5]
- An engagement score is computed from views, likes, and comments and stored in a manifest. [R4]
- Audio embeddings are 40-dim MFCC vectors (mean-pooled). [R6]
- Video embeddings are 512-dim ResNet18 features aggregated over sampled frames. [R7]

Example engagement calculation:
```python
engagement = (
    0.4 * math.log(views + 1)
    + 0.4 * (likes / (views + 1))
    + 0.2 * (comments / (views + 1))
)
```
[R4]

### 3) Modeling and Training
- A joint projection model maps video and audio embeddings into a shared 256-dim space using two MLPs with ReLU + LayerNorm. [R9]
- Training uses a contrastive objective with temperature scaling; the loss is weighted by engagement. [R8]
- Training runs for 10 epochs with Adam (lr=1e-4) and batch size 16. [R9]

Example training loop snippet:
```python
v_emb, a_emb = model(video, audio)
loss = engagement_contrastive_loss(v_emb, a_emb, engagement)
```
[R8][R9]

### 4) Inference and Serving
- The inference engine loads the trained model and precomputed audio features, then embeds a query video and computes cosine similarity for ranking. [R12]
- FastAPI exposes a `/recommend` endpoint that returns the top-k audio IDs and URLs. [R10]
- Streamlit provides a UI for upload and playback. [R11]

Example API call (curl):
```bash
curl -F "file=@sample.mp4" "http://127.0.0.1:8000/recommend?top_k=5"
```
[R10]

Example response shape:
```json
[
  {"audio_id": "abc123", "score": 0.8123, "audio_url": "http://127.0.0.1:8000/audio/abc123"}
]
```
[R10]

## Results
The repository includes evaluation scripts that compute retrieval metrics (Recall@1/5/10 and Mean Rank) using cosine similarity between video and audio embeddings. [R13] A manual inspection utility prints the top-5 audio matches for a specific video ID. [R14]

Example evaluation usage:
```bash
python evaluation/retrieval_eval.py
```
[R13]

Note: Actual metric values depend on the local dataset and are not hardcoded in the repository.

## Conclusion
The Python Audio-Video Matcher demonstrates a full pipeline for audio-video retrieval: ingesting short videos, extracting consistent audio/visual features, training a joint embedding model, and serving recommendations through an API and UI. The modular structure makes it straightforward to swap encoders, adjust loss functions, or extend evaluation. The current approach uses MFCCs and ResNet18 features with engagement-weighted contrastive learning, providing a solid baseline for audio matching in short-form content. [R1][R4][R6][R7][R8][R9][R10][R12]

## References
- [R1] Orchestrated pipeline script: `run_project.py`
- [R2] Metadata ingestion and filtering: `ingestion/fetch_metadata.py`
- [R3] Video download (yt-dlp constraints): `ingestion/download_videos.py`
- [R4] Video/audio processing and manifest: `ingestion/process_videos.py`
- [R5] Audio extraction (ffmpeg): `utils/audio_extractor.py`
- [R6] Audio features (MFCC): `utils/audio_features.py`
- [R7] Video features (ResNet18): `utils/video_features.py`
- [R8] Engagement-weighted contrastive loss: `training/loss.py`
- [R9] Joint embedding model + training loop: `models/joint_model.py`, `training/train.py`
- [R10] API server and endpoints: `api/app.py`
- [R11] Streamlit UI: `frontend/app.py`
- [R12] Inference engine: `inference/engine.py`
- [R13] Retrieval evaluation metrics: `evaluation/retrieval_eval.py`
- [R14] Manual top-k check script: `evaluation/manual_check.py`
