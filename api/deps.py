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
_detector_weights = None
_reader = None


def get_detector():
    global _detector, _detector_weights
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
        from src.detection.workbench import discover_latest_run

        latest_run = discover_latest_run(PROJECT_ROOT / "models" / "detection")
        latest_weights = latest_run / "weights" / "best.pt" if latest_run else None
        if latest_weights and latest_weights.is_file():
            configured_mtime = configured_plate_weights.stat().st_mtime if configured_plate_weights.exists() else 0
            if latest_weights.stat().st_mtime >= configured_mtime:
                configured_plate_weights = latest_weights
        if not configured_plate_weights.exists():
            return None

        character_config = config.get("character", {})
        configured_character_weights = Path(character_config.get("weights", ""))
        if configured_character_weights and not configured_character_weights.is_absolute():
            configured_character_weights = PROJECT_ROOT / configured_character_weights
        latest_character_run = discover_latest_run(PROJECT_ROOT / "models" / "character")
        latest_character_weights = latest_character_run / "weights" / "best.pt" if latest_character_run else None
        if latest_character_weights and latest_character_weights.is_file():
            configured_mtime = (
                configured_character_weights.stat().st_mtime if configured_character_weights.is_file() else 0
            )
            if latest_character_weights.stat().st_mtime >= configured_mtime:
                configured_character_weights = latest_character_weights
        character_ready = bool(character_config.get("enabled") and configured_character_weights.is_file())
        character_config["enabled"] = character_ready
        if character_ready:
            character_config["weights"] = str(configured_character_weights)

        model_weights = (configured_plate_weights, configured_character_weights if character_ready else None)
        if _detector is not None and _detector_weights == model_weights:
            return _detector
        config["plate"]["weights"] = str(configured_plate_weights)
        _detector = build_two_stage_detector(config, PROJECT_ROOT)
        _detector_weights = model_weights
        logger.info(
            "Loaded detection cascade with plate model %s and character model %s",
            configured_plate_weights,
            configured_character_weights if character_ready else "disabled",
        )
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
