import torch
import torchvision.transforms as T
from PIL import Image
import os
import numpy as np
from torchvision import models
import cv2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
_model.fc = torch.nn.Identity()
_model = _model.to(DEVICE)
_model.eval()

_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class VideoEmbedder:
    def __init__(self, device="cpu"):
        self.device = device

        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = torch.nn.Identity()  # 512-dim embedding
        model.eval()
        model.to(device)

        self.model = model

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def embed_frames(self, frame_dir, sample_rate=5):
        """
        frame_dir: directory of extracted frames
        sample_rate: take every Nth frame
        """

        frames = sorted(os.listdir(frame_dir))[::sample_rate]
        embeddings = []

        for frame in frames:
            img_path = os.path.join(frame_dir, frame)
            image = Image.open(img_path).convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)

            emb = self.model(tensor)
            embeddings.append(emb.squeeze(0))

        if len(embeddings) == 0:
            return None

        # Temporal aggregation (mean pooling)
        return torch.stack(embeddings).mean(dim=0)

def extract_video_embedding(video_path, max_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(_transform(frame))

    cap.release()

    if len(frames) == 0:
        raise RuntimeError("No frames extracted from video")

    frames = torch.stack(frames).to(DEVICE)

    with torch.no_grad():
        feats = _model(frames)
        video_emb = feats.mean(dim=0)

    return video_emb.cpu().numpy()