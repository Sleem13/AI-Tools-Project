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

    def __init__(
        self,
        detector: LicensePlateDetector | TwoStageDetector,
        reader: PlateReader,
        min_confidence: float = 0.25,
    ) -> None:
        self.detector = detector
        self.reader = reader
        self.min_confidence = min_confidence

    def _get_crops(self, image: np.ndarray):
        """Yield crops from detector, compatible with both detector types."""
        try:
            crops = self.detector.crop_plates(image, min_confidence=self.min_confidence)
        except TypeError:
            crops = self.detector.crop_plates(image)

        for crop, detection_info in crops:
            if hasattr(detection_info, "to_dict"):
                metadata = detection_info.to_dict()
            else:
                metadata = {"detection_confidence": float(detection_info)}

            detection_confidence = metadata.get("confidence", metadata.get("detection_confidence", 0.0))
            if detection_confidence < self.min_confidence:
                continue
            yield crop, metadata

    def run(self, image: np.ndarray) -> list[dict]:
        results: list[dict] = []
        for crop, metadata in self._get_crops(image):
            # OCR result
            text_ocr = self.reader.read_plate(crop)
            # Optional character detection
            text_char = None
            conf_char_avg = 0.0
            if hasattr(self.detector, "character_detector") and self.detector.character_detector is not None:
                char_results, text_char = self.detector.character_detector.recognize(crop)
                if char_results:
                    confs = [getattr(getattr(r, "detection", None), "confidence", 0.0) for r in char_results]
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
            item = dict(metadata)
            item.update(
                {
                    "plate_text": final_text,
                    "formatted_text": format_plate(final_text),
                    "confidence": confidence,
                }
            )
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
