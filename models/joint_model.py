import torch
import torch.nn as nn

class JointEmbeddingModel(nn.Module):
    def __init__(self, video_encoder, audio_encoder):
        super().__init__()
        self.video_encoder = video_encoder
        self.audio_encoder = audio_encoder

    def forward(self, video, audio):
        v_emb = self.video_encoder(video)
        a_emb = self.audio_encoder(audio)
        return v_emb, a_emb
