"""Lazy model singletons — loaded on first request, cached for the session."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("alpr_api")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DET_WEIGHTS = PROJECT_ROOT / "models" / "weights" / "detection_best.pt"
OCR_WEIGHTS = PROJECT_ROOT / "models" / "weights" / "ocr_best.pth"

_detector = None
_reader = None


def get_detector():
    global _detector
    if _detector is not None:
        return _detector
    if not DET_WEIGHTS.exists():
        return None
    try:
        from src.detection.inference import LicensePlateDetector

        _detector = LicensePlateDetector(weights_path=DET_WEIGHTS, conf_threshold=0.25)
        logger.info("Loaded detection model: %s", DET_WEIGHTS)
        return _detector
    except Exception as e:
        logger.warning("Failed to load detection model: %s", e)
        return None


def get_reader():
    global _reader
    if _reader is not None:
        return _reader
    if not OCR_WEIGHTS.exists():
        return None
    try:
        from src.ocr.inference import PlateReader

        _reader = PlateReader(weights_path=OCR_WEIGHTS)
        logger.info("Loaded OCR model: %s", OCR_WEIGHTS)
        return _reader
    except Exception as e:
        logger.warning("Failed to load OCR model: %s", e)
        return None
