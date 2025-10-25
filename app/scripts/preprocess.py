import os
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

RAW_DIR = "app/data/raw"
PROC_DIR = "app/data/processed"
FEATURE_DIR = "app/data/features"

# Ensure output directories exist
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)

def preprocess_audio(file_path):
    """Load, trim, normalize, and resample audio."""
    # Load audio (convert to mono, 16kHz)
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    
    # Trim leading and trailing silence
    y, _ = librosa.effects.trim(y, top_db=20)
    
    # Normalize loudness
    y = librosa.util.normalize(y)
    
    return y, sr

def extract_log_mel(y, sr):
    """Extract log-mel spectrogram features."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, fmax=8000
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel

def process_all_audios(limit=100):
    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".wav")]
    files = files[:limit]  # limit to 100 samples for test

    print(f"Processing {len(files)} files...")

    for file in tqdm(files):
        path = os.path.join(RAW_DIR, file)
        
        # Step 1: Preprocess
        y, sr = preprocess_audio(path)
        
        # Step 2: Save cleaned audio
        proc_path = os.path.join(PROC_DIR, file)
        sf.write(proc_path, y, sr)
        
        # Step 3: Extract features
        log_mel = extract_log_mel(y, sr)
        
        # Step 4: Save as .npy
        feature_path = os.path.join(FEATURE_DIR, file.replace(".wav", ".npy"))
        np.save(feature_path, log_mel)

    print("✅ Preprocessing complete!")

if __name__ == "__main__":
    process_all_audios()
