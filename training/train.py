import torch
from torch.utils.data import DataLoader
from datasets.av_embedding_dataset import AVEmbeddingDataset
from models.joint_model import JointEmbeddingModel
from training.loss import engagement_contrastive_loss

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = AVEmbeddingDataset("data/processed/manifest.json")
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = JointEmbeddingModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(200):
    model.train()
    total_loss = 0

    for video, audio, engagement in loader:
        video = video.to(device)
        audio = audio.to(device)
        engagement = engagement.to(device)

        v_emb, a_emb = model(video, audio)
        loss = engagement_contrastive_loss(v_emb, a_emb, engagement)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")

torch.save({
    "model_state": model.state_dict(),
    "embedding_dim": v_emb.shape[1],
    "trained_on": len(dataset),
}, "models/joint_model.pt")


# import torch
# from models.video_encoder import VideoEncoder
# from models.audio_encoder import AudioEncoder
# from models.joint_model import JointEmbeddingModel
# from training.loss import engagement_contrastive_loss
# from training.dummy_loader import get_dummy_batch
# from torch.utils.data import DataLoader
# from datasets.av_dataset import AudioVideoDataset

# samples = [
#     {
#         "video_path": "data/raw/sample.mp4",
#         "audio_path": "data/raw/sample.mpeg",
#         "engagement": 0.8
#     }
# ]

# dataset = AudioVideoDataset(samples)
# loader = DataLoader(dataset, batch_size=1, shuffle=True)

# device = "cuda" if torch.cuda.is_available() else "cpu"

# video_enc = VideoEncoder().to(device)
# audio_enc = AudioEncoder().to(device)
# model = JointEmbeddingModel(video_enc, audio_enc).to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# for epoch in range(20):
#     for video, audio, engagement in loader:
#         video = video.to(device)
#         audio = audio.to(device)
#         engagement = engagement.to(device)

#         v_emb, a_emb = model(video, audio)
#         loss = engagement_contrastive_loss(v_emb, a_emb, engagement)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         print(f"loss={loss.item():.4f}")

