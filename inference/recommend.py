import torch
import torch.nn.functional as F

def recommend_audio(video_embedding, audio_embeddings, k=5):
    scores = F.cosine_similarity(video_embedding, audio_embeddings)
    topk = torch.topk(scores, k)
    return topk.indices, topk.values
