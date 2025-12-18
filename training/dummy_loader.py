import torch

def get_dummy_batch(batch_size=4):
    video = torch.randn(batch_size, 16, 3, 224, 224)
    audio = torch.randn(batch_size, 1, 128, 400)
    engagement = torch.rand(batch_size)
    return video, audio, engagement
