"""End-to-end ALPR pipeline: detection -> OCR -> post-processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.detection.inference import LicensePlateDetector
from src.ocr.inference import PlateReader
from src.postprocessing.plate_formatter import format_plate


@dataclass
class ALPRResult:
    """Result of running the full ALPR pipeline on one image."""

    image_path: str | Path
    detections: list[dict]  # [{bbox, confidence, plate_text, formatted_text}]
    processing_time_ms: float


class ALPRPipeline:
    """Chain detection -> OCR -> post-processing for end-to-end inference."""

    def __init__(
        self,
        detector: LicensePlateDetector,
        reader: PlateReader,
        min_confidence: float = 0.25,
    ) -> None:
        self.detector = detector
        self.reader = reader
        self.min_confidence = min_confidence

    def run(self, image: np.ndarray) -> list[dict]:
        """Run full ALPR on a single image.

        Args:
            image: BGR numpy array.

        Returns:
            List of detection dicts with keys:
                bbox, confidence, plate_text, formatted_text
        """
        crops = self.detector.crop_plates(image, min_confidence=self.min_confidence)
        results: list[dict] = []
        for crop, conf in crops:
            text = self.reader.read_plate(crop)
            formatted = format_plate(text)
            results.append({
                "plate_text": text,
                "formatted_text": formatted,
                "confidence": conf,
            })
        return results

    def run_path(self, image_path: str | Path) -> ALPRResult:
        """Run full ALPR on an image file."""
        import time
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read: {image_path}")

        t0 = time.perf_counter()
        detections = self.run(image)
        elapsed = (time.perf_counter() - t0) * 1000.0

        return ALPRResult(
            image_path=Path(image_path),
            detections=detections,
            processing_time_ms=elapsed,
        )



