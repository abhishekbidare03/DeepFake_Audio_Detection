from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import audio_routes

app = FastAPI(title="Deepfake Audio Detection API")

# Enable CORS for frontend access (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(audio_routes.router)


@app.get("/")
def home():
    return {"message": "Welcome to Deepfake Audio Detection API"}
