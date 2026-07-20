"""ALPR API — FastAPI backend for the React frontend."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import datasets, detection, pipeline, training

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(
    title="ALPR Pipeline API",
    version="1.0.0",
    description="Backend API for Egyptian License Plate Recognition pipeline",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detection.router, prefix="/api")
app.include_router(pipeline.router, prefix="/api")
app.include_router(datasets.router, prefix="/api")
app.include_router(training.router, prefix="/api")


@app.get("/api/health")
def health():
    from api.deps import get_detector, get_reader

    det = get_detector()
    ocr = get_reader()
    return {
        "status": "ok",
        "models_loaded": {
            "detection": det is not None,
            "ocr": ocr is not None,
        },
    }
