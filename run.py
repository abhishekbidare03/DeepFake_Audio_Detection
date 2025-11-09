"""
Run script that ensures correct Python path and starts the FastAPI server
"""
import os
import sys
from pathlib import Path
import uvicorn

# Get the absolute path to the project root
project_root = Path(__file__).resolve().parent
print(f"Project root: {project_root}")

# Add project root to Python path
sys.path.insert(0, str(project_root))

# Set the working directory to project root
os.chdir(str(project_root))

if __name__ == "__main__":
    print("Starting server...")
    print(f"Python path: {sys.path}")
    
    # Run the FastAPI application
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )