from fastapi import FastAPI
from app.routes import audio_routes

app = FastAPI(title="Deepfake Audio Detection API")

# Register routes
app.include_router(audio_routes.router)

@app.get("/")
def home():
    return {"message": "Welcome to Deepfake Audio Detection API"}
