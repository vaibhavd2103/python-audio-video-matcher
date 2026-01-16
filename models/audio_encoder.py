# import torch
# import torch.nn as nn

# class AudioEncoder(nn.Module):
#     def __init__(self, embed_dim=512):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(1, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((1, 1))
#         )
#         self.fc = nn.Linear(64, embed_dim)

#     def forward(self, x):
#         x = self.conv(x)
#         x = x.view(x.size(0), -1)
#         return self.fc(x)
