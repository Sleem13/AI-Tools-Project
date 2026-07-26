"""End-to-end vehicle -> plate -> text ALPR inference pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from src.detection.inference import TwoStageDetector
from src.ocr.inference import PlateReader
from src.postprocessing.plate_formatter import format_plate


@dataclass
class ALPRResult:
    image_path: str | Path
    detections: list[dict]
    processing_time_ms: float


class ALPRPipeline:
    """Chain YOLO11 vehicle/plate detection with the configured text reader."""

    def __init__(self, detector: TwoStageDetector, reader: PlateReader) -> None:
        self.detector = detector
        self.reader = reader

    def run(self, image: np.ndarray) -> list[dict]:
        results: list[dict] = []
        for crop, cascade_detection in self.detector.crop_plates(image):
            text = self.reader.read_plate(crop)
            item = cascade_detection.to_dict()
            item.update({"plate_text": text, "formatted_text": format_plate(text)})
            results.append(item)
        return results

    def run_path(self, image_path: str | Path) -> ALPRResult:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read: {image_path}")
        started = time.perf_counter()
        detections = self.run(image)
        return ALPRResult(
            image_path=Path(image_path),
            detections=detections,
            processing_time_ms=(time.perf_counter() - started) * 1000.0,
        )
