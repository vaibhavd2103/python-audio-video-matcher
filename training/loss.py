import torch
import torch.nn.functional as F

def engagement_contrastive_loss(video_emb, audio_emb, engagement):
    sim = F.cosine_similarity(video_emb, audio_emb)
    loss = -torch.log(torch.sigmoid(sim))
    return (loss * engagement).mean()
