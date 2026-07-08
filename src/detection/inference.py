"""Inference wrapper for the trained license plate detection model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class DetectionResult:
    """A single detected license plate."""

    bbox: tuple[float, float, float, float]  # xyxy (normalized or pixel)
    confidence: float
    class_id: int
    class_name: str


class LicensePlateDetector:
    """Load a trained YOLO model and run inference on images.

    Handles image pre-processing, inference, and post-processing
    (NMS, confidence filtering).  The actual model is loaded lazily.
    """

    def __init__(
        self,
        weights_path: str | Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ) -> None:
        self.weights_path = Path(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(str(self.weights_path))

    def predict(self, image: np.ndarray) -> list[DetectionResult]:
        """Run detection on a single BGR image.

        Args:
            image: BGR numpy array (H, W, 3).

        Returns:
            List of DetectionResult, one per detected plate.
        """
        self._load_model()
        results = self._model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        return self._parse_results(results)

    def predict_path(self, image_path: str | Path) -> list[DetectionResult]:
        """Convenience: read image from disk and detect."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return self.predict(image)

    def _parse_results(self, results) -> list[DetectionResult]:
        detections: list[DetectionResult] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = r.names.get(cls_id, str(cls_id))
                detections.append(
                    DetectionResult(
                        bbox=tuple(xyxy),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=cls_name,
                    )
                )
        return detections

    def crop_plates(self, image: np.ndarray, min_confidence: float = 0.25) -> list[tuple[np.ndarray, float]]:
        """Detect plates and return cropped images.

        Args:
            image: BGR image.
            min_confidence: Minimum confidence to keep a detection.

        Returns:
            List of (plate_crop, confidence) tuples.
        """
        detections = self.predict(image)
        crops: list[tuple[np.ndarray, float]] = []
        h, w = image.shape[:2]
        for det in detections:
            if det.confidence < min_confidence:
                continue
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                crop = image[y1:y2, x1:x2]
                crops.append((crop, det.confidence))
        return crops
