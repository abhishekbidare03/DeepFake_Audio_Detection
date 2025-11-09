import os
import hashlib
import time
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.services.inference_service import predict as run_predict
from app.utils.validators import max_file_size_ok, sanitize_filename

def is_valid_audio_filename(filename: str) -> bool:
    """Check if the file extension is supported."""
    return filename.lower().endswith(('.wav', '.flac', '.mp3'))
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
    print(f"Received file: {file.filename}, content_type: {file.content_type}")
    
    if not is_valid_audio_filename(file.filename):
        raise HTTPException(status_code=400, detail="Only .wav, .mp3 and .flac files are supported")

    content = await file.read()
    if not max_file_size_ok(content):
        raise HTTPException(status_code=413, detail="File too large")

    # Compute hash for caching
    file_hash = hashlib.sha256(content).hexdigest()
    cached = prediction_cache.get(file_hash)
    if cached is not None:
        # Map cached fields to response format
        confidence_fake_pct = round(float(cached.get('prob_fake', 0.0)) * 100, 2)
        confidence_real_pct = round(float(cached.get('prob_real', 0.0)) * 100, 2)
        label_text = cached.get('label', 'Unknown')
        reason = ''
        if confidence_fake_pct >= 60:
            reason = 'Detected unusual spectral patterns consistent with synthetic audio.'
            
        return JSONResponse(content={
            'confidence_fake': confidence_fake_pct,
            'confidence_real': confidence_real_pct,
            'label': label_text,
            'reason': reason,
            'cached': True
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
            # Use the best performing model
            model_path = 'models/combined_final_testacc_61.95.pth'
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}. Please ensure the model file is in the models directory.")
            result = run_predict(temp_path, model_path=model_path)

        # Cache the result (store simplified response)
        cached_payload = {
            'label': result.get('label'),
            'prob_fake': result.get('prob_fake'),
            'prob_real': result.get('prob_real')
        }
        prediction_cache.set(file_hash, cached_payload)

        # Map to required fields
        confidence_fake_pct = round(float(result.get('prob_fake', 0.0)) * 100, 2)
        confidence_real_pct = round(float(result.get('prob_real', 0.0)) * 100, 2)
        label_text = result.get('label', 'Unknown')
        reason = ''
        if confidence_fake_pct >= 60:
            reason = 'Detected unusual spectral patterns consistent with synthetic audio.'

        response = {
            'confidence_fake': confidence_fake_pct,
            'confidence_real': confidence_real_pct,
            'label': label_text,
            'reason': reason,
            'inference_time_sec': round(t.elapsed, 4),
            'cached': False
        }
        return JSONResponse(content=response)

    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model file not found: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"Error processing audio: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)  # Log the full error
        raise HTTPException(status_code=500, detail=error_detail)
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
