import numpy as np
import librosa

def extract_audio_features(audio_path: str) -> np.ndarray:
    """
    Extract MFCC-based audio embedding.
    Returns a fixed-size vector.
    """

    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40
    )

    # Mean pooling over time → (40,)
    features = mfcc.mean(axis=1)

    return features.astype("float32")
