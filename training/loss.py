# import torch
# import torch.nn.functional as F

# def engagement_contrastive_loss(v, a, engagement, temperature=0.07):
#     v = F.normalize(v, dim=1)
#     a = F.normalize(a, dim=1)

#     logits = torch.matmul(v, a.T) / temperature
#     labels = torch.arange(len(v), device=v.device)

#     # Per-sample losses so we can weight by engagement.
#     loss_v_to_a = F.cross_entropy(logits, labels, reduction="none")
#     loss_a_to_v = F.cross_entropy(logits.T, labels, reduction="none")

#     # Engagement weighting (higher engagement => higher importance)
#     weights = engagement / (engagement.mean() + 1e-8)
#     weighted = (loss_v_to_a + loss_a_to_v) * weights

#     return weighted.mean()


import torch
import torch.nn.functional as F

# def engagement_contrastive_loss(video_emb, audio_emb, engagement):
#     sim = F.cosine_similarity(video_emb, audio_emb)
#     loss = -torch.log(torch.sigmoid(sim))
#     return (loss * engagement).mean()

def engagement_contrastive_loss(video_emb, audio_emb, engagement, temperature=0.07):
    """
    video_emb: [B, D]
    audio_emb: [B, D]
    engagement: [B]  (normalized 0–1)
    """

    video_emb = F.normalize(video_emb, dim=1)
    audio_emb = F.normalize(audio_emb, dim=1)

    logits = video_emb @ audio_emb.T / temperature
    labels = torch.arange(video_emb.size(0), device=video_emb.device)

    loss = F.cross_entropy(logits, labels, reduction="none")

    # engagement-weighted supervision
    loss = loss * engagement

    return loss.mean()
