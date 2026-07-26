"""Lazy model singletons — loaded on first request, cached for the session."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger("alpr_api")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DET_WEIGHTS = PROJECT_ROOT / "models" / "weights" / "detection_best.pt"
TWO_STAGE_CONFIG = PROJECT_ROOT / "configs" / "model" / "two_stage.yaml"
OCR_WEIGHTS = PROJECT_ROOT / "models" / "weights" / "ocr_best.pth"

_detector = None
_reader = None


def get_detector():
    global _detector
    if _detector is not None:
        return _detector
    if not TWO_STAGE_CONFIG.exists():
        return None
    try:
        from src.detection.inference import build_two_stage_detector

        with TWO_STAGE_CONFIG.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
        configured_plate_weights = Path(config["plate"]["weights"])
        if not configured_plate_weights.is_absolute():
            configured_plate_weights = PROJECT_ROOT / configured_plate_weights
        if not configured_plate_weights.exists() and DET_WEIGHTS.exists():
            configured_plate_weights = DET_WEIGHTS
        if not configured_plate_weights.exists():
            return None
        config["plate"]["weights"] = str(configured_plate_weights)
        _detector = build_two_stage_detector(config, PROJECT_ROOT)
        logger.info("Loaded two-stage detector with plate model: %s", configured_plate_weights)
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
