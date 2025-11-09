"""
Test script to start the FastAPI server with uvicorn.
"""
import uvicorn
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    # Run the FastAPI application with uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )