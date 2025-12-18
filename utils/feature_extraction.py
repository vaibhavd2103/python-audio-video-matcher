import cv2
import librosa
import torch
import numpy as np

def extract_video_frames(path, num_frames=16):
    cap = cv2.VideoCapture(path)
    frames = []

    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)

    cap.release()

    frames = np.stack(frames)
    frames = torch.tensor(frames).permute(0, 3, 1, 2).float() / 255.0
    return frames


def extract_mel_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel = librosa.power_to_db(mel)
    return torch.tensor(mel).unsqueeze(0)
