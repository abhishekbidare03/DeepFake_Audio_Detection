import sys
from pathlib import Path
import logging
import asyncio
from fastapi.testclient import TestClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from main import app

def test_api():
    """Test all API endpoints"""
    client = TestClient(app)
    
    # Test root endpoint
    logger.info("Testing root endpoint...")
    response = client.get("/")
    assert response.status_code == 200
    root_data = response.json()
    logger.info(f"Root endpoint response: {root_data}")
    
    # Test health check
    logger.info("\nTesting health check...")
    response = client.get("/api/v1/audio/health")
    assert response.status_code == 200
    health_data = response.json()
    logger.info(f"Health check response: {health_data}")
    
    # Test model info
    logger.info("\nTesting model info...")
    response = client.get("/api/v1/audio/model-info")
    assert response.status_code == 200
    model_info = response.json()
    logger.info(f"Model info response: {model_info}")
    
    # Test single file prediction
    logger.info("\nTesting single file prediction...")
    demo_file = project_root / "Demo_samples" / "demo.wav"
    
    with open(demo_file, "rb") as f:
        files = {"file": ("demo.wav", f, "audio/wav")}
        response = client.post("/api/v1/audio/predict", files=files)
        assert response.status_code == 200
        prediction = response.json()
        logger.info(f"Single prediction response: {prediction}")
    
    # Test batch prediction with 2 files
    logger.info("\nTesting batch prediction...")
    demo_files = list((project_root / "Demo_samples").glob("*.wav"))[:2]
    
    files = []
    for demo_file in demo_files:
        with open(demo_file, "rb") as f:
            files.append(("files", (demo_file.name, f, "audio/wav")))
    
    response = client.post("/api/v1/audio/batch-predict", files=files)
    assert response.status_code == 200
    batch_results = response.json()
    logger.info(f"Batch prediction response: {batch_results}")
    
    logger.info("\nAll API tests passed successfully!")

if __name__ == "__main__":
    test_api()