# datasets/av_dataset.py
import torch
from torch.utils.data import Dataset
from utils.feature_extraction import extract_video_frames, extract_mel_spectrogram

class AudioVideoDataset(Dataset):
    def __init__(self, samples):
        """
        samples = [
            {
              "video_path": "...mp4",
              "audio_path": "...wav",
              "engagement": float
            }
        ]
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        video = extract_video_frames(sample["video_path"])
        audio = extract_mel_spectrogram(sample["audio_path"])
        engagement = torch.tensor(sample["engagement"]).float()

        return video, audio, engagement
