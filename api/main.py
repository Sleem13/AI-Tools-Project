"""ALPR API — FastAPI backend for the React frontend."""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import datasets, detection, pipeline, training

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(
    title="ALPR Pipeline API",
    version="1.0.0",
    description="Backend API for Egyptian License Plate Recognition pipeline",
)

default_origins = "http://localhost:5173,http://127.0.0.1:5173"
cors_origins = [
    origin.strip()
    for origin in os.getenv("ALPR_CORS_ORIGINS", default_origins).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detection.router, prefix="/api")
app.include_router(pipeline.router, prefix="/api")
app.include_router(datasets.router, prefix="/api")
app.include_router(training.router, prefix="/api")


@app.on_event("startup")
def startup_event():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 8000))
    except OSError:
        logging.getLogger(__name__).warning(
            "Port 8000 may still be in use by a previous process. "
            "Kill it with: netstat -ano | findstr :8000 && taskkill /PID <PID> /F"
        )
    finally:
        sock.close()


@app.on_event("shutdown")
def shutdown_event():
    from api.training_jobs import training_job_manager
    from api.video_jobs import video_job_manager

    training_job_manager.cleanup()
    video_job_manager.cleanup()


@app.get("/api/health")
def health():
    from api.deps import get_detector, get_reader

    det = get_detector()
    ocr = get_reader()
    ocr_path = str(getattr(ocr, "weights_path", "")) if ocr is not None else None
    return {
        "status": "ok",
        "models_loaded": {
            "detection": det is not None,
            "character": bool(det is not None and getattr(det, "character_detector", None) is not None),
            "ocr": ocr is not None,
        },
        "model_sources": {
            "ocr": getattr(ocr, "source", None) if ocr is not None else None,
            "ocr_path": ocr_path,
        },
    }
