import os
import librosa
import soundfile as sf
import numpy as np

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

def extract_features(y, sr):
    """
    Extract mel spectrogram features from audio signal
    Args:
        y: Audio signal
        sr: Sample rate
    Returns:
        log_mel: Log-mel spectrogram
    """
    mel = librosa.feature.melspectrogram(
        y=y, 
        sr=sr,
        n_mels=128,
        fmax=8000
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
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
