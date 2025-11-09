from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import shutil
from pathlib import Path
import uuid
from app.models.ml_model import get_model
from app.services.audio_processor import AudioProcessor

router = APIRouter()
audio_processor = AudioProcessor()

# Create temp directory for uploaded files
UPLOAD_DIR = Path("data/raw")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    """
    Predict if audio is fake or real
    """
    # Validate file type
    if not file.filename.endswith(('.wav', '.mp3', '.flac', '.ogg')):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Supported: .wav, .mp3, .flac, .ogg"
        )
    
    # Save uploaded file temporarily
    temp_file = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
    
    try:
        with temp_file.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Validate audio
        if not audio_processor.validate_audio(str(temp_file)):
            raise HTTPException(status_code=400, detail="Invalid audio file")
        
        # Extract features
        features = audio_processor.extract_features(str(temp_file))
        
        # Get model and predict
        model = get_model()
        prediction_score = model.predict(features)
        
        # Determine result
        is_fake = prediction_score > 0.5
        confidence = prediction_score if is_fake else (1 - prediction_score)
        
        return JSONResponse({
            "status": "success",
            "prediction": "FAKE" if is_fake else "REAL",
            "confidence": round(float(confidence * 100), 2),
            "score": round(float(prediction_score), 4)
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        # Clean up temp file
        if temp_file.exists():
            temp_file.unlink()

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Audio detection API is running"}