"""Detection module for ALPR license plate detection.

Provides:
  - LicensePlateDataset: YOLO-format dataset loader with augmentations
  - train_yolo: programmatic YOLOv8 training entry point
  - LicensePlateDetector: inference wrapper loading a trained YOLO model
"""

from src.detection.data import LicensePlateDataset, create_dataset_yaml
from src.detection.inference import LicensePlateDetector, DetectionResult
from src.detection.trainer import train_yolo

__all__ = [
    "LicensePlateDataset",
    "create_dataset_yaml",
    "LicensePlateDetector",
    "DetectionResult",
    "train_yolo",
]
