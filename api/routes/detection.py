"""POST /api/detect — upload image, run detection + OCR."""

from __future__ import annotations

import time
from io import BytesIO

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, UploadFile

from api.deps import get_detector, get_reader

router = APIRouter()


@router.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
):
    content = await file.read()
    arr = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return {"error": "Could not decode image", "detections": [], "processing_time_ms": 0}

    detector = get_detector()
    reader = get_reader()

    if detector is None:
        return {
            "error": "Detection model not loaded",
            "detections": [],
            "processing_time_ms": 0,
        }

    t0 = time.perf_counter()

    detector.conf_threshold = conf
    crops = detector.crop_plates(image, min_confidence=conf)

    detections = []
    for crop, det_conf in crops:
        text = ""
        formatted = ""
        if reader is not None:
            text = reader.read_plate(crop)
            from src.postprocessing.plate_formatter import format_plate

            formatted = format_plate(text)

        h, w = image.shape[:2]
        dets = detector.predict(image)
        # Get bbox from the matching detection
        bbox = [0, 0, 0, 0]
        for d in dets:
            if abs(d.confidence - det_conf) < 0.01:
                bbox = list(d.bbox)
                break

        detections.append({
            "bbox": bbox,
            "confidence": det_conf,
            "plate_text": text,
            "formatted_text": formatted,
        })

    elapsed = (time.perf_counter() - t0) * 1000.0

    return {"detections": detections, "processing_time_ms": round(elapsed, 1)}
