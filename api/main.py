from fastapi import FastAPI, UploadFile
import torch

app = FastAPI()

@app.post("/recommend")
async def recommend(video: UploadFile):
    # extract features → embed → retrieve
    return {"recommended_audio_ids": ["a1", "a2", "a3"]}
