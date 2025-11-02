import os
import hashlib
import time
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.services.inference_service import predict as run_predict
from app.utils.validators import is_valid_audio_filename, max_file_size_ok
from app.utils.monitoring import timeit
from app.utils.cache_manager import prediction_cache

router = APIRouter(prefix="/audio", tags=["Audio Detection"])


@router.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Save uploaded file to data/raw (used by teammates/front-end)."""
    if not is_valid_audio_filename(file.filename):
        raise HTTPException(status_code=400, detail="Only .wav and .flac files are supported")

    content = await file.read()
    if not max_file_size_ok(content):
        raise HTTPException(status_code=413, detail="File too large")

    save_dir = os.path.join("data", "raw")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file.filename)

    try:
        with open(save_path, "wb") as f:
            f.write(content)
        return {"filename": file.filename, "status": "File uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")


@router.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    """Predict single uploaded audio file and return label + probabilities.
    Uses a small in-memory cache to avoid repeated inference on the same bytes.
    """
    if not is_valid_audio_filename(file.filename):
        raise HTTPException(status_code=400, detail="Only .wav and .flac files are supported")

    content = await file.read()
    if not max_file_size_ok(content):
        raise HTTPException(status_code=413, detail="File too large")

    # Compute hash for caching
    file_hash = hashlib.sha256(content).hexdigest()
    cached = prediction_cache.get(file_hash)
    if cached is not None:
        return JSONResponse(content={
            "prediction": cached['label'],
            "probabilities": cached['probabilities'],
            "cached": True
        })

    # Save to temp file because librosa expects a path
    temp_dir = os.path.join("data", "temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{int(time.time()*1000)}_{file.filename}")
    try:
        with open(temp_path, "wb") as f:
            f.write(content)

        # Run prediction and measure time
        with timeit() as t:
            result = run_predict(temp_path)

        # Cache the result
        prediction_cache.set(file_hash, result)

        response = {
            "prediction": result['label'],
            "probabilities": result['probabilities'],
            "inference_time_sec": round(t.elapsed, 4),
            "cached": False
        }
        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
