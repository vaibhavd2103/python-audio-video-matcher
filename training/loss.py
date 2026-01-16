import torch
import torch.nn.functional as F

def engagement_contrastive_loss(v, a, engagement, temperature=0.07):
    v = F.normalize(v, dim=1)
    a = F.normalize(a, dim=1)

    logits = torch.matmul(v, a.T) / temperature
    labels = torch.arange(len(v), device=v.device)

    base_loss = F.cross_entropy(logits, labels)

    # Engagement weighting (higher engagement → higher importance)
    weights = engagement / engagement.mean()
    weighted_loss = (base_loss * weights.mean())

    return weighted_loss


# import torch
# import torch.nn.functional as F

# def engagement_contrastive_loss(video_emb, audio_emb, engagement):
#     sim = F.cosine_similarity(video_emb, audio_emb)
#     loss = -torch.log(torch.sigmoid(sim))
#     return (loss * engagement).mean()
