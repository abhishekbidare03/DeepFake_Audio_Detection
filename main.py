# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging

# import router
from app.routes import audio_routes

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Deepfake Audio Detection API")

# Development CORS (in production restrict origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# include router (audio_routes defines its own prefix)
app.include_router(audio_routes.router)

# mount frontend after routers so API routes take precedence
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


@app.on_event("startup")
async def on_startup():
    logger.info("Starting Deepfake Audio Detection API")


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Shutting down Deepfake Audio Detection API")
