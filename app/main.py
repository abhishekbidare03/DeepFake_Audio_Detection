from fastapi import FastAPI
import uvicorn

from app.routes import audio_routes

app = FastAPI(title="DeepFake Audio Detector API")

app.include_router(audio_routes.router)


def start():
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == '__main__':
    start()
