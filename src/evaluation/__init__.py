"""Evaluation metrics and end-to-end ALPR pipeline evaluation.

Provides:
  - DetectionMetrics: mAP, precision, recall for license plate detection
  - OCRMetrics: character error rate (CER), word error rate (WER)
  - ALPREvaluator: combined evaluation across detection + OCR
"""

from src.evaluation.metrics import ALPREvaluator, DetectionMetrics, OCRMetrics
from src.evaluation.pipeline import ALPRPipeline

__all__ = [
    "ALPREvaluator",
    "ALPRPipeline",
    "DetectionMetrics",
    "OCRMetrics",
]
