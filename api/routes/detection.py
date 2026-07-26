"""POST /api/detect - run vehicle -> plate detection and optional OCR."""

from __future__ import annotations

import time
from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, UploadFile

from api.deps import get_detector, get_reader

router = APIRouter()


@router.post("/detect")
async def detect(
    file: Annotated[UploadFile, File()],
    conf: Annotated[float, Form()] = 0.25,
):
    content = await file.read()
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return {"error": "Could not decode image", "detections": [], "processing_time_ms": 0}

    detector = get_detector()
    reader = get_reader()
    if detector is None:
        return {
            "error": "Plate model not loaded; train stage two or configure its weights",
            "detections": [],
            "processing_time_ms": 0,
        }

    started = time.perf_counter()
    detector.plate_detector.conf_threshold = conf
    detections = []
    for crop, cascade_detection in detector.crop_plates(image):
        text = reader.read_plate(crop) if reader is not None else ""
        formatted = ""
        if text:
            from src.postprocessing.plate_formatter import format_plate

            formatted = format_plate(text)
        item = cascade_detection.to_dict()
        item.update({"plate_text": text, "formatted_text": formatted})
        detections.append(item)

    elapsed = (time.perf_counter() - started) * 1000.0
    return {"detections": detections, "processing_time_ms": round(elapsed, 1)}
