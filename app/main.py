from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.routes import audio_routes
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path

app = FastAPI(title="DeepFake Audio Detector API")

# Enable CORS for demo frontend. In production, set a specific origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(audio_routes.router)

# Mount API routes first
app.include_router(audio_routes.router)

# Create necessary directories
os.makedirs("data/temp", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Ensure default model exists
DEFAULT_MODEL = 'models/best_model.pth'
if not os.path.exists(DEFAULT_MODEL):
    raise RuntimeError(f"Default model file not found: {DEFAULT_MODEL}. Please add a model file.")

# Serve frontend static files from the Frontend directory
FRONTEND_DIR = os.path.join(Path(__file__).resolve().parents[1], 'Frontend')
if os.path.exists(FRONTEND_DIR):
    # Mount frontend at root
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    @app.get("/")
    def home():
        return {"message": "DeepFake Audio Detection API is running"}


def start():
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == '__main__':
    start()
