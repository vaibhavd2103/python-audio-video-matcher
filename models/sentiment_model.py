# import torch
# import torch.nn as nn

# class SentimentHead(nn.Module):
#     def __init__(self, input_dim=512, num_classes=5):
#         super().__init__()
#         self.fc = nn.Linear(input_dim, num_classes)

#     def forward(self, x):
#         return torch.softmax(self.fc(x), dim=1)
