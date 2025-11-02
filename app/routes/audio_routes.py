import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import torch
import numpy as np
from app.utils.audio_utils import load_and_process_audio, extract_features
from app.models.cnn_model import AudioCNN

router = APIRouter(prefix="/audio", tags=["Audio Detection"])

# Load model (do this at startup)
model = None
try:
    model = AudioCNN()
    model.load_state_dict(torch.load("app/models/model.pt"))
    model.eval()
except Exception as e:
    print(f"Warning: Could not load model: {str(e)}")

def is_valid_audio(filename: str) -> bool:
    """Check if file has valid audio extension"""
    return filename.lower().endswith(('.wav', '.flac'))

@router.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    if not is_valid_audio(file.filename):
        raise HTTPException(status_code=400, detail="Only .wav and .flac files are supported")
    
    # Save the uploaded file
    save_path = os.path.join("data/raw", file.filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        with open(save_path, "wb") as buffer:
            buffer.write(await file.read())
        return {"filename": file.filename, "status": "File uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

@router.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    if not is_valid_audio(file.filename):
        raise HTTPException(status_code=400, detail="Only .wav and .flac files are supported")
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Save temporary file
        temp_path = os.path.join("data/temp", file.filename)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Process audio
        y, sr = load_and_process_audio(temp_path)
        features = extract_features(y, sr)
        
        # Prepare for model
        features = np.expand_dims(features, axis=0)  # Add batch dimension
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        
        # Predict
        with torch.no_grad():
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1).item()
            confidence = probs[0][prediction].item()
        
        # Clean up
        os.remove(temp_path)
        
        return JSONResponse(content={
            "prediction": "Real" if prediction == 0 else "Fake",
            "confidence": float(confidence),
            "filename": file.filename
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
