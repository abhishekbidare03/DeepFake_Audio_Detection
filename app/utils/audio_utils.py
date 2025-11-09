import os
import torch
import librosa
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset

def load_and_process_audio(file_path, target_sr=16000):
    """
    Load and process audio file (supports both .wav and .flac)
    Args:
        file_path: Path to audio file (.wav or .flac)
        target_sr: Target sample rate (default: 16000 Hz)
    Returns:
        y: Audio signal
        sr: Sample rate
    """
    # Load audio file
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    
    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=20)
    
    # Normalize
    y = librosa.util.normalize(y)
    
    return y, sr

def extract_features(y, sr, fixed_length=64):
    """
    Extract mel spectrogram features from audio signal with fixed length
    Args:
        y: Audio signal
        sr: Sample rate
        fixed_length: Fixed number of time frames for the spectrogram
    Returns:
        log_mel: Log-mel spectrogram with fixed length
    """
    # Calculate hop length to achieve approximately fixed_length frames
    n_fft = 2048
    hop_length = int(len(y) / fixed_length)
    hop_length = max(hop_length, 512)  # Ensure minimum hop length
    
    mel = librosa.feature.melspectrogram(
        y=y, 
        sr=sr,
        n_mels=128,
        n_fft=n_fft,
        hop_length=hop_length,
        fmax=8000
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    
    # Adjust length to fixed_length
    if log_mel.shape[1] > fixed_length:
        log_mel = log_mel[:, :fixed_length]
    elif log_mel.shape[1] < fixed_length:
        pad_width = fixed_length - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
    
    return log_mel

def save_audio(y, sr, output_path):
    """
    Save audio file (automatically handles .wav or .flac based on extension)
    Args:
        y: Audio signal
        sr: Sample rate
        output_path: Path to save audio file
    """
    sf.write(output_path, y, sr)

def get_audio_info(file_path):
    """
    Get audio file information
    Args:
        file_path: Path to audio file
    Returns:
        duration: Duration in seconds
        sr: Sample rate
        channels: Number of channels
    """
    info = sf.info(file_path)
    return {
        'duration': info.duration,
        'sample_rate': info.samplerate,
        'channels': info.channels
    }

class AudioDataset(Dataset):
    def __init__(self, data_dir):
        """
        Initialize audio dataset
        Args:
            data_dir: Directory containing audio files in real/ and fake/ subdirectories
        """
        self.data_dir = data_dir
        self.samples = []
        
        # Load real samples (label 0)
        real_dir = os.path.join(data_dir, 'real')
        if os.path.exists(real_dir):
            for file in os.listdir(real_dir):
                if file.endswith(('.wav', '.flac')):
                    self.samples.append((os.path.join(real_dir, file), 0))
        
        # Load fake samples (label 1)
        fake_dir = os.path.join(data_dir, 'fake')
        if os.path.exists(fake_dir):
            for file in os.listdir(fake_dir):
                if file.endswith(('.wav', '.flac')):
                    self.samples.append((os.path.join(fake_dir, file), 1))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        
        # Load and process audio
        y, sr = load_and_process_audio(audio_path)
        
        # Extract features
        features = extract_features(y, sr)
        
        # Convert to tensor and add channel dimension
        features = torch.FloatTensor(features).unsqueeze(0)
        
        return features, label
