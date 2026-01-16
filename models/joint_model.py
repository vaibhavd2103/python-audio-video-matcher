import torch
import torch.nn as nn

class JointEmbeddingModel(nn.Module):
    def __init__(self, video_dim=512, audio_dim=40, embed_dim=256):
        super().__init__()

        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
        )

        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, video_emb, audio_emb):
        v = self.video_proj(video_emb)
        a = self.audio_proj(audio_emb)
        return v, a


# import torch
# import torch.nn as nn

# class JointEmbeddingModel(nn.Module):
#     def __init__(self, video_encoder, audio_encoder):
#         super().__init__()
#         self.video_encoder = video_encoder
#         self.audio_encoder = audio_encoder

#     def forward(self, video, audio):
#         v_emb = self.video_encoder(video)
#         a_emb = self.audio_encoder(audio)
#         return v_emb, a_emb
