"""YOLO11 vehicle-to-plate detection and training."""

from src.detection.data import LicensePlateDataset, create_dataset_yaml
from src.detection.inference import (
    COCO_VEHICLE_CLASS_IDS,
    DetectionResult,
    LicensePlateDetector,
    TwoStageDetection,
    TwoStageDetector,
    VehicleDetector,
    YOLODetector,
    build_two_stage_detector,
)
from src.detection.trainer import train_detector, train_yolo

__all__ = [
    "COCO_VEHICLE_CLASS_IDS",
    "DetectionResult",
    "LicensePlateDataset",
    "LicensePlateDetector",
    "TwoStageDetection",
    "TwoStageDetector",
    "VehicleDetector",
    "YOLODetector",
    "build_two_stage_detector",
    "create_dataset_yaml",
    "train_detector",
    "train_yolo",
]
