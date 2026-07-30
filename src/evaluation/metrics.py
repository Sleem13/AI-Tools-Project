"""Evaluation metrics for detection and OCR stages."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DetectionMetrics:
    """Aggregated detection metrics."""

    precision: float = 0.0
    recall: float = 0.0
    mAP50: float = 0.0
    mAP50_95: float = 0.0
    n_images: int = 0
    n_detections: int = 0
    n_ground_truths: int = 0

    def summary(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "mAP50": round(self.mAP50, 4),
            "mAP50_95": round(self.mAP50_95, 4),
            "n_images": self.n_images,
            "n_detections": self.n_detections,
            "n_ground_truths": self.n_ground_truths,
        }


@dataclass
class OCRMetrics:
    """Aggregated OCR metrics."""

    cer: float = 0.0
    wer: float = 0.0
    accuracy: float = 0.0
    n_samples: int = 0
    n_errors: int = 0
    per_char_errors: dict[str, int] = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "cer": round(self.cer, 4),
            "wer": round(self.wer, 4),
            "accuracy": round(self.accuracy, 4),
            "n_samples": self.n_samples,
            "n_errors": self.n_errors,
        }


class ALPREvaluator:
    """Combined ALPR evaluation tracking detection + OCR metrics."""

    def __init__(self) -> None:
        self.detection = DetectionMetrics()
        self.ocr = OCRMetrics()

    def update_detection(self, metrics: DetectionMetrics) -> None:
        self.detection = metrics

    def update_ocr(self, metrics: OCRMetrics) -> None:
        self.ocr = metrics

    def summary(self) -> dict:
        return {
            "detection": self.detection.summary(),
            "ocr": self.ocr.summary(),
        }
