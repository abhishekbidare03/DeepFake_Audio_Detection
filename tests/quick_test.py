"""Quick test script to verify API endpoints"""
import requests
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API base URL
BASE_URL = "http://127.0.0.1:8000"

def test_endpoints():
    """Test all API endpoints"""
    try:
        # Test root endpoint
        logger.info("Testing root endpoint...")
        response = requests.get(f"{BASE_URL}/")
        assert response.status_code == 200
        logger.info("Root endpoint OK")
        
        # Test health check
        logger.info("\nTesting health endpoint...")
        response = requests.get(f"{BASE_URL}/api/v1/audio/health")
        assert response.status_code == 200
        logger.info("Health endpoint OK")
        logger.info(f"Health status: {response.json()}")
        
        # Test predict endpoint with demo file
        logger.info("\nTesting predict endpoint...")
        demo_file = Path(__file__).parents[1] / "Demo_samples" / "demo.wav"
        
        if not demo_file.exists():
            logger.error(f"Demo file not found: {demo_file}")
            return
        
        with open(demo_file, "rb") as f:
            files = {"file": ("demo.wav", f, "audio/wav")}
            response = requests.post(
                f"{BASE_URL}/api/v1/audio/predict",
                files=files
            )
            
            assert response.status_code == 200
            logger.info("Predict endpoint OK")
            logger.info(f"Prediction result: {response.json()}")
        
        logger.info("\nAll endpoint tests passed!")
        
    except requests.exceptions.ConnectionError:
        logger.error(f"Could not connect to API at {BASE_URL}")
        logger.error("Make sure the server is running (python run_server.py)")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_endpoints()