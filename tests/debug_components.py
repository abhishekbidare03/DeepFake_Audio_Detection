"""
Debug script to test the FastAPI application components
"""
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def test_model_loading():
    """Test model loading without the API"""
    try:
        from app.models.fake_detector import create_model
        import torch
        
        logger.info("Creating model...")
        model = create_model()
        
        logger.info("Loading weights...")
        model_path = project_root / "models" / "best_model.pth"
        state = torch.load(str(model_path))
        model.load_state_dict(state)
        
        logger.info("Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def test_audio_utils():
    """Test audio utilities"""
    try:
        from app.utils.audio_utils import AudioUtils
        import librosa
        import numpy as np
        
        audio_utils = AudioUtils()
        
        # Test with a demo file
        demo_file = project_root / "Demo_samples" / "demo.wav"
        
        logger.info(f"Loading audio file: {demo_file}")
        audio, sr = audio_utils.load_audio(str(demo_file))
        
        logger.info("Processing audio...")
        audio = audio_utils.normalize_audio(audio)
        audio = audio_utils.trim_silence(audio)
        
        logger.info("Extracting features...")
        mel = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr,
            n_mels=128,
            fmax=8000
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        
        logger.info(f"Feature shape: {log_mel.shape}")
        return True
        
    except Exception as e:
        logger.error(f"Error in audio processing: {str(e)}")
        return False

def run_tests():
    """Run all tests"""
    logger.info("Starting tests...")
    
    logger.info("\nTesting model loading:")
    if test_model_loading():
        logger.info("✓ Model loading test passed")
    else:
        logger.error("✗ Model loading test failed")
    
    logger.info("\nTesting audio utils:")
    if test_audio_utils():
        logger.info("✓ Audio utils test passed")
    else:
        logger.error("✗ Audio utils test failed")

if __name__ == "__main__":
    run_tests()