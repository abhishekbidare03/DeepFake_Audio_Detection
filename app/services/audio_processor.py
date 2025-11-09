import librosa
import numpy as np
from pathlib import Path


class AudioProcessor:
    def __init__(self, sr: int = 16000):
        self.sr = sr

    def extract_features(self, audio_path: str) -> np.ndarray:
        """Extract a fixed-size 1D feature vector from an audio file.

        Returns a 1D numpy array (e.g. [mfcc40_means..., spec_centroid_mean, spec_rolloff_mean, zcr_mean]).
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sr, mono=True)

        # Extract features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)

        # Compute summary statistics
        mfccs_mean = np.mean(mfccs, axis=1)              # shape (40,)
        sc_mean = np.array([np.mean(spectral_centroid)]) # shape (1,)
        sr_mean = np.array([np.mean(spectral_rolloff)])  # shape (1,)
        zcr_mean = np.array([np.mean(zero_crossing_rate)])

        # Concatenate into a single vector
        features = np.hstack([mfccs_mean, sc_mean, sr_mean, zcr_mean])

        return features

    def validate_audio(self, audio_path: str) -> bool:
        """Validate if file is a readable audio file by attempting a short load."""
        try:
            librosa.load(audio_path, sr=self.sr, duration=1, mono=True)
            return True
        except Exception:
            return False