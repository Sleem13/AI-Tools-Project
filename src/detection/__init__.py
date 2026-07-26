"""YOLO11 vehicle-to-plate detection and training."""

from src.detection.data import LicensePlateDataset, create_dataset_yaml
from src.detection.inference import (
    CHARACTER_GLYPHS,
    COCO_VEHICLE_CLASS_IDS,
    CharacterDetector,
    CharacterResult,
    DetectionResult,
    LicensePlateDetector,
    TwoStageDetection,
    TwoStageDetector,
    VehicleDetector,
    YOLODetector,
    build_two_stage_detector,
    order_and_decode_characters,
)
from src.detection.trainer import train_detector, train_yolo

__all__ = [
    "CHARACTER_GLYPHS",
    "COCO_VEHICLE_CLASS_IDS",
    "CharacterDetector",
    "CharacterResult",
    "DetectionResult",
    "LicensePlateDataset",
    "LicensePlateDetector",
    "TwoStageDetection",
    "TwoStageDetector",
    "VehicleDetector",
    "YOLODetector",
    "build_two_stage_detector",
    "create_dataset_yaml",
    "order_and_decode_characters",
    "train_detector",
    "train_yolo",
]
