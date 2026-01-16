# import torch
# import torch.nn as nn
# from torchvision.models import resnet18

# class VideoEncoder(nn.Module):
#     def __init__(self, embed_dim=512):
#         super().__init__()
#         base = resnet18(weights="DEFAULT")
#         self.backbone = nn.Sequential(*list(base.children())[:-1])
#         self.fc = nn.Linear(512, embed_dim)

#     def forward(self, x):
#         # x: [B, T, C, H, W]
#         b, t, c, h, w = x.shape
#         x = x.view(b * t, c, h, w)
#         feats = self.backbone(x).squeeze(-1).squeeze(-1)
#         feats = feats.view(b, t, -1).mean(dim=1)
#         return self.fc(feats)
