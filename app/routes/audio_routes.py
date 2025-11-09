# app/routes/audio_routes.py
import os
import time
import hashlib
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# your inference function
from app.services.inference_service import predict as run_predict

# utilities you'd been using
from app.utils.validators import max_file_size_ok, sanitize_filename
from app.utils.monitoring import timeit
from app.utils.cache_manager import prediction_cache

router = APIRouter(prefix="/api/v1/audio", tags=["Audio Detection"])

# Directories
BASE_DIR = Path("data")
RAW_DIR = BASE_DIR / "raw"
TEMP_DIR = BASE_DIR / "temp"
RAW_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Model path — keep your existing trained model file name
MODEL_PATH = Path("models/combined_final_testacc_61.95.pth")


def is_valid_audio_filename(filename: str) -> bool:
    return filename.lower().endswith(('.wav', '.flac', '.mp3', '.m4a', '.ogg'))


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_exists": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH)
    }


@router.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Optional endpoint: saves file to data/raw"""
    if not is_valid_audio_filename(file.filename):
        raise HTTPException(status_code=400, detail="Only .wav, .flac, .mp3, .m4a, .ogg files supported")

    content = await file.read()
    if not max_file_size_ok(content):
        raise HTTPException(status_code=413, detail="File too large")

    safe_name = sanitize_filename(file.filename)
    save_path = RAW_DIR / safe_name
    try:
        with open(save_path, "wb") as f:
            f.write(content)
        return {"filename": safe_name, "status": "File uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")


@router.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    """
    Predict single uploaded audio file and return label + probabilities.
    Uses in-memory cache via prediction_cache (if available).
    """
    # Basic checks
    if not is_valid_audio_filename(file.filename):
        raise HTTPException(status_code=400, detail="Only .wav, .mp3, .flac, .m4a, .ogg files are supported")

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

    # Save to temporary file because many audio libs expect a path
    ts = int(time.time() * 1000)
    safe_name = sanitize_filename(file.filename)
    temp_path = TEMP_DIR / f"{ts}_{safe_name}"

    try:
        with open(temp_path, "wb") as f:
            f.write(content)

        # Run prediction and measure time
        with timeit() as t:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Please ensure model is present.")
            # run_predict expected signature: predict(file_path, model_path=...)
            result = run_predict(str(temp_path), model_path=str(MODEL_PATH))

        # Cache simplified payload to avoid re-running heavy inference
        cached_payload = {
            'label': result.get('label'),
            'prob_fake': result.get('prob_fake'),
            'prob_real': result.get('prob_real')
        }
        prediction_cache.set(file_hash, cached_payload)

        # Map model result to response keys used by frontend
        # Accept model outputs like prob_fake (0..1) or prob_real or label
        prob_fake = None
        if isinstance(result, dict):
            if 'prob_fake' in result:
                prob_fake = float(result.get('prob_fake', 0.0))
            elif 'prob_real' in result:
                prob_fake = 1.0 - float(result.get('prob_real', 0.0))
            elif 'probabilities' in result and isinstance(result['probabilities'], dict):
                probs = result['probabilities']
                prob_fake = float(probs.get('fake', probs.get('FAKE', probs.get('Fake', 0.0))))
            elif 'score' in result:
                # some models return a single score; assume it's P(fake) when name ambiguous
                prob_fake = float(result.get('score', 0.0))
            # else leave prob_fake None

        # Fallbacks if prob_fake not provided
        label = result.get('label') if isinstance(result, dict) else None
        if prob_fake is None:
            if label:
                if label.lower() in ('fake', 'ai-generated', 'synthetic', 'synth'):
                    prob_fake = 1.0
                elif label.lower() in ('real', 'authentic', 'genuine'):
                    prob_fake = 0.0
            else:
                prob_fake = 0.5

        confidence_fake_pct = round(float(prob_fake) * 100.0, 2)
        confidence_real_pct = round(100.0 - confidence_fake_pct, 2)
        label_text = label or ("FAKE" if confidence_fake_pct >= 50.0 else "REAL")
        reason_text = result.get('reason') if isinstance(result, dict) else ''

        response = {
            'confidence_fake': confidence_fake_pct,
            'confidence_real': confidence_real_pct,
            'label': label_text,
            'reason': reason_text,
            'inference_time_sec': round(t.elapsed, 4),
            'cached': False,
            'raw': result
        }

        return JSONResponse(content=response)

    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"Error processing audio: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        # For dev: return trace in detail; in production sanitize message
        raise HTTPException(status_code=500, detail=error_detail)
    finally:
        # Always try to remove temp file
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass
