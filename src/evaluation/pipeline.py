"""End-to-end vehicle -> plate -> text ALPR inference pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from src.detection.inference import LicensePlateDetector, TwoStageDetector
from src.ocr.inference import PlateReader
from src.postprocessing.plate_formatter import format_plate


@dataclass
class ALPRResult:
    image_path: str | Path
    detections: list[dict]
    processing_time_ms: float


class ALPRPipeline:
    """Chain license plate detection with OCR and optional character detection fusion."""

    def __init__(self, detector: LicensePlateDetector | TwoStageDetector, reader: PlateReader) -> None:
        self.detector = detector
        self.reader = reader

    def _get_crops(self, image: np.ndarray):
        """Yield crops from detector, compatible with both detector types."""
        for crop, _ in self.detector.crop_plates(image):
            yield crop

    def run(self, image: np.ndarray) -> list[dict]:
        results: list[dict] = []
        for crop in self._get_crops(image):
            # OCR result
            text_ocr = self.reader.read_plate(crop)
            # Optional character detection
            text_char = None
            conf_char_avg = 0.0
            if hasattr(self.detector, "character_detector") and self.detector.character_detector is not None:
                char_results, text_char = self.detector.character_detector.recognize(crop)
                if char_results:
                    confs = [getattr(r, "confidence", 0.0) for r in char_results]
                    if confs:
                        conf_char_avg = sum(confs) / len(confs)
            # Fusion: use character detection if confident enough, else OCR
            if text_char and conf_char_avg > 0.5:
                final_text = text_char
                confidence = conf_char_avg
            else:
                final_text = text_ocr
                confidence = 1.0  # placeholder; OCR confidence not used
            # Build a minimal detection dict; we could try to get metadata but keep simple.
            item = {}
            item.update({"plate_text": final_text, "formatted_text": format_plate(final_text)})
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