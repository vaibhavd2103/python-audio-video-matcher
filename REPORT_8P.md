# Python Audio-Video Matcher: Technical Report (Expanded)

## Abstract
This report presents a Python-based system that ingests short-form videos, extracts audio and visual features, learns a shared audio-video embedding space, and serves top-k audio recommendations for a query video. The pipeline integrates metadata collection, media download, preprocessing, feature extraction, model training, evaluation, and a FastAPI + Streamlit interface. The model aligns audio and video embeddings using an engagement-weighted contrastive objective and returns the most similar audio tracks. The report provides technical details, file-level references, and concrete examples for each major component.

## Introduction
Short-form video platforms rely on matching audio to visuals for discovery, remixing, and trend analysis. An audio-video matcher supports tasks such as recommending audio tracks that fit a visual clip, clustering content by soundtrack, and assisting creators in selecting audio for new videos. This project implements a complete end-to-end pipeline in Python that goes from raw data collection to a deployable inference service. The system is orchestrated by a single runner script that executes ingestion, preprocessing, feature extraction, training, and serving steps in order. [R1]

Example (end-to-end run):
```bash
python run_project.py
```

## Problem Statement
Given a collection of short videos with audio, build a system that can recommend the most relevant audio tracks for a new video clip. The system must (1) ingest and filter short videos that contain audio, (2) compute consistent audio and visual embeddings, (3) learn a joint embedding space where matching audio-video pairs are close, and (4) serve top-k audio recommendations for a new video. [R1][R2][R3][R4]

Example requirement mapping:
- Ingestion and filtering: `ingestion/fetch_metadata.py` (duration and status filters). [R2]
- Media download: `ingestion/download_videos.py` (audio+video constraints). [R3]
- Feature extraction: `ingestion/extract_audio_features.py`, `ingestion/extract_video_features.py`. [R6][R7]
- Serving: `api/app.py`, `frontend/app.py`. [R10][R11]

## Objectives
- Collect metadata for short videos and filter for valid, public, short-duration items. [R2]
- Download short videos with both audio and video tracks. [R3]
- Extract frames and audio, then compute fixed-length audio and video embeddings. [R4][R5][R6][R7]
- Train a joint embedding model with an engagement-aware contrastive loss. [R8][R9]
- Provide an API and UI for inference and audio playback. [R10][R11][R12]

Example objective alignment (audio features):
```python
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
features = mfcc.mean(axis=1)
```
[R6]

## Background and Related Work
Audio-visual representation learning typically uses CNN-based visual encoders and spectral or MFCC-based audio embeddings, mapped into a shared representation space. Contrastive learning is a common objective for aligning pairs because it encourages correct pairs to be close while pushing mismatches apart. Retrieval evaluation (Recall@K, Mean Rank) is a standard protocol for measuring how often the correct audio is ranked within the top results.

Examples in this codebase:
- Visual features via ResNet18 frame embeddings with temporal mean pooling. [R7]
- Audio features via MFCCs with mean pooling. [R6]
- Retrieval evaluation using cosine similarity and recall@k. [R13]

## System Architecture
The system is organized into five layers:
1) Ingestion (metadata, download) -> produces raw videos and metadata.
2) Processing (frames, audio) -> produces frames and WAV audio, plus a manifest.
3) Feature extraction -> produces audio and video embeddings.
4) Training -> learns a joint embedding model.
5) Serving -> exposes API and UI for recommendations.

The codebase mirrors this layout through dedicated modules and folders: `ingestion/`, `utils/`, `training/`, `models/`, `inference/`, `api/`, `frontend/`, and `evaluation/`. [R1]

Example orchestrator steps:
```python
run_command("python -m ingestion.fetch_metadata")
run_command("python -m ingestion.download_videos")
run_command("python -m ingestion.process_videos")
run_command("python -m ingestion.extract_audio_features")
run_command("python -m ingestion.extract_video_features")
run_command("python -m training.train")
```
[R1]

## Data Sources and Metadata Schema
The pipeline uses the YouTube Data API to search short-form videos and retrieve statistics. Metadata includes views, likes, and comments, which are later used for engagement weighting. The metadata is stored as JSON in `data/raw/metadata/shorts_metadata.json`. [R2]

Example metadata record:
```json
{
  "video_id": "abc123",
  "views": 10234,
  "likes": 830,
  "comments": 72
}
```
[R2]

Filtering logic example:
```python
if status.get("uploadStatus") != "processed":
    continue
if status.get("privacyStatus") != "public":
    continue
if duration == 0 or duration > 60:
    continue
```
[R2]

Configuration example (environment variable used by the API client):
```env
YOUTUBE_API_KEY=your_key_here
```
[R2]

## Methodology

### 1) Data Collection and Ingestion
- Metadata is fetched using a search query such as "songs shorts" and restricted to short-duration videos (<= 60 seconds). [R2]
- Videos are downloaded using `yt-dlp` with audio+video format constraints to avoid silent media. [R3]
- The downloader enforces timeouts and retry policies to avoid hanging requests. [R3]

Example download command pattern:
```bash
yt-dlp -f "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]" --merge-output-format mp4 ...
```
[R3]

Example downloader safeguards:
```python
"--socket-timeout", "10",
"--retries", "3",
"--fragment-retries", "3",
```
[R3]

### 2) Processing (Frames + Audio)
Frames are sampled from each video at a fixed FPS with a maximum cap, and audio is extracted to a mono 22.05 kHz WAV file using ffmpeg. [R4][R5]

Example frame extraction signature:
```python
def extract_frames(video_path, output_dir, fps=2, max_frames=30) -> int:
    ...
```
[R4]

Example audio extraction call:
```python
extract_audio(video_path, audio_out)
```
[R5]

Frame sampling rationale example:
- With fps=2 and max_frames=30, at most 30 frames represent a video.
- This bounds compute and ensures consistent processing time per video. [R4]

### 3) Engagement Scoring and Manifest
The system computes an engagement score from views, likes, and comments to approximate popularity and to weight training. A manifest JSON file stores per-video paths and engagement values. [R4]

Example engagement calculation:
```python
engagement = (
    0.4 * math.log(views + 1)
    + 0.4 * (likes / (views + 1))
    + 0.2 * (comments / (views + 1))
)
```
[R4]

Example manifest entry:
```json
{
  "video_id": "abc123",
  "video_path": "data/raw/videos/abc123.mp4",
  "frames_dir": "data/processed/frames/abc123",
  "audio_path": "data/processed/audio/abc123.wav",
  "num_frames": 24,
  "engagement": 1.92
}
```
[R4]

### 4) Audio Feature Extraction
Audio features use MFCCs with mean pooling to produce a fixed-length vector. This reduces variable-length audio to a compact representation suitable for retrieval. [R6]

Example audio feature pipeline:
```python
y, sr = librosa.load(audio_path, sr=22050, mono=True)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
features = mfcc.mean(axis=1)
```
[R6]

Output shape example:
- `mfcc` shape: (40, T)
- `features` shape: (40,)

### 5) Video Feature Extraction
Video features are computed by sampling frames and embedding each frame through a ResNet18 backbone. The per-frame features are mean-pooled to produce a 512-dimensional video embedding. [R7]

Example video embedding step:
```python
embeddings.append(emb.squeeze(0))
return torch.stack(embeddings).mean(dim=0)
```
[R7]

Embedding dimension example:
- ResNet18 output dimension: 512
- Final video embedding: 512-dim vector. [R7]

### 6) Joint Embedding Model
The joint model projects audio and video embeddings into a shared 256-dimensional space using two MLPs with ReLU and LayerNorm. [R9]

Model projection example:
```python
self.video_proj = nn.Sequential(
    nn.Linear(video_dim, embed_dim),
    nn.ReLU(),
    nn.LayerNorm(embed_dim),
)
```
[R9]

Audio projection example:
```python
self.audio_proj = nn.Sequential(
    nn.Linear(audio_dim, embed_dim),
    nn.ReLU(),
    nn.LayerNorm(embed_dim),
)
```
[R9]

### 7) Loss Function and Training
The training objective is a temperature-scaled contrastive loss on normalized embeddings, weighted by engagement. [R8]

Loss example:
```python
logits = torch.matmul(v, a.T) / temperature
base_loss = F.cross_entropy(logits, labels)
weights = engagement / engagement.mean()
weighted_loss = (base_loss * weights.mean())
```
[R8]

Training configuration example:
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(10):
    ...
```
[R9]

Batching example:
```python
loader = DataLoader(dataset, batch_size=16, shuffle=True)
```
[R9]

### 8) Inference and Retrieval
During inference, the system loads the trained model and the precomputed audio features, embeds the query video, and computes cosine similarity scores to rank audio candidates. [R12]

Similarity scoring example:
```python
v_emb = torch.nn.functional.normalize(v_emb, dim=1)
a_emb = torch.nn.functional.normalize(a_emb, dim=1)
score = torch.sum(v_emb * a_emb).item()
```
[R12]

Top-k selection example:
```python
results.sort(key=lambda x: x[1], reverse=True)
return results[:top_k]
```
[R12]

## Evaluation
The repository includes two evaluation utilities:
- Retrieval evaluation script: computes Recall@1/5/10 and Mean Rank across all paired items. [R13]
- Manual top-k inspection: prints the top-5 audios for a selected video ID. [R14]

Example evaluation run:
```bash
python evaluation/retrieval_eval.py
```
[R13]

Example metric definitions (conceptual):
- Recall@K: fraction of queries whose correct audio is in the top K.
- Mean Rank: average rank position of the correct audio.

These metrics align with common audio-video retrieval evaluation practices and are implemented via cosine similarity in the codebase. [R13]

## Results
The evaluation scripts report retrieval metrics based on the local dataset and embeddings. Actual values depend on the number of videos processed, sampling settings, and model training. The design supports quick regression checks after modifying feature extraction or the training objective.

Example output format:
```
Recall@1  : 0.123
Recall@5  : 0.372
Recall@10 : 0.515
Mean Rank : 23.84
```
[R13]

Note: The numeric values above are placeholders showing the format; real metrics come from running the evaluation script locally.

## Deployment and Interfaces
### API (FastAPI)
The API exposes a `/recommend` endpoint that accepts an uploaded video file and returns a ranked list of audio IDs and URLs. [R10]

Example request:
```bash
curl -F "file=@sample.mp4" "http://127.0.0.1:8000/recommend?top_k=5"
```
[R10]

Example response:
```json
[
  {"audio_id": "abc123", "score": 0.8123, "audio_url": "http://127.0.0.1:8000/audio/abc123"}
]
```
[R10]

API handler example:
```python
@app.post("/recommend")
async def recommend(file: UploadFile = File(...), top_k: int = 5):
    ...
```
[R10]

### UI (Streamlit)
The UI allows a user to upload a video, request recommendations, and play the returned audio files. [R11]

Example UI flow:
1) Upload a video file.
2) Select top-k in the slider.
3) Click "Recommend Audio" to show results and play audio. [R11]

Streamlit interaction example:
```python
if st.button("Recommend Audio"):
    response = requests.post(API_URL, files=files, params={"top_k": top_k})
```
[R11]

## Implementation Details and Configuration
### Dependencies
Key dependencies are listed in `requirements.txt`, including PyTorch, torchvision, librosa, OpenCV, FastAPI, and Streamlit. [R15]

Example installation:
```bash
pip install -r requirements.txt
```
[R15]

### Runtime Pipeline
The main pipeline is executed by `run_project.py` which sequentially triggers ingestion, processing, feature extraction, and training. [R1]

Example pipeline call:
```bash
python run_project.py
```
[R1]

### File and Directory Layout
The project follows a modular layout with `ingestion/`, `training/`, `models/`, `utils/`, `evaluation/`, `api/`, and `frontend/` directories, with data stored under `data/`. [R1][R2][R3][R4]

Example data structure:
```
python-audio-video-matcher/
  data/
    raw/
      videos/
      metadata/
    processed/
      frames/
      audio/
      audio_features/
      video_features/
      manifest.json
```
[R4]

### Error Handling and Skips
The ingestion and processing steps include basic error handling and filtering. Examples include skipping videos with missing audio, filtering out entries with zero frames, and removing incomplete downloads. [R3][R4][R5]

Example skip logic:
```python
if not has_audio or not os.path.exists(audio_out):
    continue
```
[R4]

## Data Flow Walkthrough
This section provides a concrete data flow walkthrough using the file paths and artifacts produced by the pipeline. The examples mirror the actual directory structure and formats used in the codebase. [R1][R2][R3][R4][R6][R7][R9][R12]

Step 1: Metadata ingestion writes JSON to `data/raw/metadata/shorts_metadata.json`. [R2]\n
Step 2: Downloaded videos are stored under `data/raw/videos/` as MP4 files. [R3]\n
Step 3: Processing extracts frames into `data/processed/frames/<video_id>/` and audio into `data/processed/audio/<video_id>.wav`, and writes a manifest to `data/processed/manifest.json`. [R4][R5]\n
Step 4: Audio features are saved as NumPy arrays in `data/processed/audio_features/<video_id>.npy`. [R6]\n
Step 5: Video features are saved as PyTorch tensors in `data/processed/video_features/<video_id>.pt`. [R7]\n
Step 6: Training reads the manifest and the feature directories and produces a model checkpoint at `models/joint_model.pt`. [R9]\n
Step 7: Inference loads the model and audio feature cache, embeds a query video, and computes similarity scores to return top-k matches. [R12]\n

Example artifact names:\n
```text
data/raw/metadata/shorts_metadata.json
data/raw/videos/abc123.mp4
data/processed/frames/abc123/frame_000.jpg
data/processed/audio/abc123.wav
data/processed/audio_features/abc123.npy
data/processed/video_features/abc123.pt
models/joint_model.pt
```\n
[R2][R3][R4][R6][R7][R9]

## Algorithm Summary (Pseudo-code)
The following pseudo-code summarizes the pipeline at a high level. It reflects the control flow in the runner script and the internal modules. [R1][R4][R6][R7][R9][R12]

```text
for each video in search_results:
    download video with audio
    extract frames and audio
    compute engagement score
    save manifest entry

for each manifest entry:
    compute audio embedding (MFCC mean)
    compute video embedding (ResNet mean)

train joint model with contrastive loss (engagement weighted)

for query video:
    compute video embedding
    score against all audio embeddings
    return top-k audio matches
```\n
[R1][R4][R6][R7][R8][R9][R12]

## Hyperparameters and Defaults
The system uses a set of hard-coded defaults spread across modules. The following list summarizes these defaults and where they are defined in code. [R4][R6][R7][R8][R9][R12]

- Frame sampling: fps=2, max_frames=30. [R4]\n
- Audio extraction: mono, 22050 Hz sample rate. [R5][R6]\n
- Audio features: n_mfcc=40. [R6]\n
- Video embedding: ResNet18 output dimension 512. [R7]\n
- Joint embedding dimension: 256. [R9]\n
- Contrastive temperature: 0.07. [R8]\n
- Training: batch_size=16, epochs=10, lr=1e-4. [R9]\n
- Inference: top_k default 5. [R10][R12]\n

Example definitions in code:\n
```python
def extract_frames(..., fps: int = 2, max_frames: int = 30) -> int:
    ...
```\n
[R4]

## Storage Formats and Artifacts
The pipeline writes multiple artifacts in standard formats to allow reuse and inspection. These formats are defined by the code and can be inspected with standard tools. [R2][R3][R4][R6][R7][R9]

- Metadata: JSON (`shorts_metadata.json`). [R2]\n
- Raw videos: MP4 files (`data/raw/videos/<video_id>.mp4`). [R3]\n
- Frames: JPG images (`data/processed/frames/<video_id>/frame_*.jpg`). [R4]\n
- Audio: WAV files (`data/processed/audio/<video_id>.wav`). [R5]\n
- Audio embeddings: NumPy `.npy` arrays (`data/processed/audio_features/<video_id>.npy`). [R6]\n
- Video embeddings: PyTorch `.pt` tensors (`data/processed/video_features/<video_id>.pt`). [R7]\n
- Model checkpoint: PyTorch `.pt` file (`models/joint_model.pt`). [R9]\n

Example load calls:\n
```python
audio_emb = np.load(os.path.join(AUDIO_DIR, f"{aid}.npy"))
video_emb = torch.load(os.path.join(VIDEO_DIR, f"{vid}.pt"))
```\n
[R13]

## Reproducibility and Determinism
The training script does not set explicit random seeds, so the model training results may vary between runs. This is typical for PyTorch training unless deterministic settings and seeds are enforced. [R9]\n

Suggested reproducibility additions (proposed):\n
- Set `torch.manual_seed` and `numpy.random.seed` at the start of training.\n
- Store configuration parameters (fps, n_mfcc, embed_dim) alongside the model checkpoint.\n
- Log dataset size and version in the training output.\n

## Quality Assurance and Testing
There are no automated unit tests in the repository; validation is done via retrieval evaluation and manual checks. [R13][R14]\n

Example manual inspection:\n
```bash
python evaluation/manual_check.py
```\n
[R14]

Example retrieval evaluation:\n
```bash
python evaluation/retrieval_eval.py
```\n
[R13]

Suggested testing improvements (proposed):\n
- Unit tests for frame extraction and audio extraction output shapes.\n
- Integration tests for the full pipeline on a small sample set.\n
- API tests verifying `/recommend` response schema.\n

## Complexity and Performance Considerations
The pipeline is designed for clarity and modularity but has several computational hotspots:
- Video feature extraction is the most expensive step because it runs a CNN per frame. [R7]
- Audio feature extraction is relatively lightweight due to MFCCs and mean pooling. [R6]
- Inference scales linearly with the number of audio embeddings since it computes similarity against all candidates. [R12]

Example linear scan in inference:
```python
for audio_id, audio_tensor in audio_features.items():
    v_emb, a_emb = model(video_tensor, audio_tensor.unsqueeze(0))
    ...
```
[R12]

Potential optimizations (proposed):
- Pre-normalize audio embeddings for faster similarity computation.
- Use approximate nearest neighbor search for large datasets.

## Use Cases and Example Scenarios
The system can support multiple workflows where audio recommendations are useful. These scenarios are driven by the capabilities implemented in the API and inference engine. [R10][R12]

Use case 1: Creator assistance\n
Creators can upload a new video clip and receive a ranked list of audio tracks that match the clip's visual style. The API provides a top-k list with scores, and the UI allows immediate playback. [R10][R11]

Example workflow:\n
```text
Upload clip -> /recommend -> receive ranked audio list -> preview audio
```\n
[R10][R11]

Use case 2: Dataset exploration\n
Analysts can embed a video and search for audio tracks that are most similar, which is a form of content-based retrieval. The retrieval output provides a simple way to inspect alignment between modalities. [R12][R13]

Example analysis script:\n
```python
results = recommend_audio(\"data/raw/sample.mp4\", top_k=5)
```\n
[R12]

Use case 3: Validation and debugging\n
During model development, it is useful to manually inspect whether the recommended audios are reasonable for a given visual clip. The `manual_check.py` script provides this manual inspection pathway. [R14]

Example manual check:\n
```bash
python evaluation/manual_check.py
```\n
[R14]

## Operational Considerations
This section focuses on how the system behaves in a local deployment and what operational constraints arise from the implementation choices. The points below are grounded in the actual code paths. [R1][R10][R12]

Startup sequence:\n
1) Run the pipeline to generate features and train the model.\n
2) Start the API server with `uvicorn`.\n
3) Launch the Streamlit UI for user interaction. [R1][R10][R11]

Example commands:\n
```bash
python run_project.py
uvicorn api.app:app --reload
streamlit run frontend/app.py
```\n
[R1][R10][R11]

Model loading behavior:\n
The API loads the model checkpoint once at startup and keeps it in memory for subsequent requests. This reduces per-request latency but requires that the checkpoint file exists before the API is started. [R10]\n

Example checkpoint load:\n
```python
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt[\"model_state\"])
```\n
[R10]

Audio feature cache:\n
The API loads all audio embeddings into memory at startup, which speeds up similarity scoring at the cost of memory usage. This design is suitable for small to medium datasets; for larger datasets, a database or vector index would be more appropriate. [R10][R12]\n

Example cache construction:\n
```python
audio_embeddings[vid] = torch.tensor(np.load(...)).to(DEVICE)
```\n
[R10]

Request handling:\n
The `/recommend` endpoint saves the uploaded video to disk, then uses the inference engine to score and rank audio tracks. The returned results include both IDs and direct audio URLs for playback. [R10][R12]\n

Example response fields:\n
```json
{\"audio_id\": \"abc123\", \"score\": 0.8123, \"audio_url\": \"http://127.0.0.1:8000/audio/abc123\"}
```\n
[R10]

## Limitations
- The model uses relatively lightweight feature extractors (MFCC and ResNet18), which may limit accuracy compared to larger audio and video encoders. [R6][R7]
- Engagement weighting is a proxy signal and may not correlate perfectly with audio-video semantic alignment. [R4][R8]
- The system assumes audio is present and extractable for each video, which can fail for silent or corrupted files. [R3][R5]
- Retrieval evaluation is based on exact ID matching; it does not measure perceptual similarity or user preference. [R13]

## Ethical and Legal Considerations
- Using platform APIs and downloaded content requires compliance with the platform's terms of service and copyright rules. The pipeline expects a valid API key and uses public videos only. [R2][R3]
- Audio recommendation systems can amplify trends or bias; engagement-driven weighting may favor already popular content. [R4]
- Any deployment should ensure proper licensing for media and avoid distributing copyrighted audio without permission.

## Future Work
- Replace MFCCs with learned audio embeddings (e.g., CNN or transformer-based) and compare retrieval metrics. [R6]
- Swap ResNet18 with a stronger video encoder or add temporal modeling (3D CNN or transformer). [R7]
- Add offline evaluation datasets with explicit ground truth for audio-video matching. [R13]
- Introduce negative sampling strategies or multi-similarity losses for better separation. [R8]
- Add caching and batching for faster inference throughput. [R12]

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
- [R15] Dependencies list: `requirements.txt`
