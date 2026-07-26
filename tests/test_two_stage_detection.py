"""Tests for the vehicle-to-license-plate YOLO cascade."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from src.detection.inference import (
    COCO_VEHICLE_CLASS_IDS,
    DetectionResult,
    TwoStageDetector,
    YOLODetector,
    build_two_stage_detector,
)


class StubDetector:
    def __init__(self, batches: list[list[DetectionResult]]) -> None:
        self.batches = iter(batches)
        self.image_shapes: list[tuple[int, ...]] = []

    def predict(self, image: np.ndarray) -> list[DetectionResult]:
        self.image_shapes.append(image.shape)
        return next(self.batches)


def _detection(
    bbox: tuple[float, float, float, float],
    confidence: float,
    class_id: int = 0,
    class_name: str = "license_plate",
) -> DetectionResult:
    return DetectionResult(bbox=bbox, confidence=confidence, class_id=class_id, class_name=class_name)


class FakeUltralyticsModel:
    def __init__(self) -> None:
        self.predict_args = {}

    def predict(self, **kwargs):
        self.predict_args = kwargs
        boxes = [
            SimpleNamespace(
                xyxy=np.array([[1.0, 2.0, 30.0, 40.0]]),
                conf=np.array([0.85]),
                cls=np.array([2]),
            )
        ]
        return [SimpleNamespace(boxes=boxes, names={2: "car"})]


def test_yolo_adapter_passes_class_filter_and_parses_results() -> None:
    detector = YOLODetector("yolo11n.pt", class_ids={2}, device="cpu", max_detections=25)
    model = FakeUltralyticsModel()
    detector._model = model

    results = detector.predict(np.zeros((50, 50, 3), dtype=np.uint8))

    assert results == [_detection((1.0, 2.0, 30.0, 40.0), 0.85, 2, "car")]
    assert model.predict_args["classes"] == [2]
    assert model.predict_args["max_det"] == 25
    assert model.predict_args["verbose"] is False


def test_yolo_adapter_rejects_empty_images() -> None:
    detector = YOLODetector("yolo11n.pt")

    with pytest.raises(ValueError, match="non-empty"):
        detector.predict(np.array([]))


def test_maps_plate_bbox_back_to_full_image() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    vehicle = _detection((20, 10, 120, 70), 0.9, 2, "car")
    local_plate = _detection((10, 20, 50, 40), 0.8)
    vehicle_detector = StubDetector([[vehicle]])
    plate_detector = StubDetector([[local_plate]])
    detector = TwoStageDetector(vehicle_detector, plate_detector, vehicle_padding=0)

    results = detector.predict(image)

    assert len(results) == 1
    assert results[0].plate.bbox == (30.0, 30.0, 70.0, 50.0)
    assert results[0].plate_bbox_in_vehicle == (10.0, 20.0, 50.0, 40.0)
    assert results[0].combined_confidence == pytest.approx(0.72)
    assert plate_detector.image_shapes == [(60, 100, 3)]


def test_expands_vehicle_crop_with_padding_and_clips_to_image() -> None:
    image = np.zeros((50, 80, 3), dtype=np.uint8)
    vehicle = _detection((0, 0, 40, 30), 0.9, 2, "car")
    plate_detector = StubDetector([[]])
    detector = TwoStageDetector(StubDetector([[vehicle]]), plate_detector, vehicle_padding=0.25)

    detector.predict(image)

    assert plate_detector.image_shapes == [(38, 50, 3)]


def test_deduplicates_plate_candidates_from_overlapping_vehicles() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    vehicles = [
        _detection((0, 0, 80, 80), 0.8, 2, "car"),
        _detection((0, 0, 80, 80), 0.95, 7, "truck"),
    ]
    plate_batches = [[_detection((10, 10, 30, 25), 0.9)], [_detection((11, 10, 31, 25), 0.9)]]
    detector = TwoStageDetector(StubDetector([vehicles]), StubDetector(plate_batches), vehicle_padding=0)

    results = detector.predict(image)

    assert len(results) == 1
    assert results[0].vehicle.class_name == "truck"


def test_full_image_fallback_can_preserve_plate_recall() -> None:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    plate = _detection((5, 6, 20, 18), 0.7)
    detector = TwoStageDetector(
        StubDetector([[]]),
        StubDetector([[plate]]),
        fallback_to_full_image=True,
    )

    results = detector.predict(image)

    assert len(results) == 1
    assert results[0].vehicle.class_name == "full_image"
    assert results[0].plate.bbox == plate.bbox


def test_crop_plates_returns_global_plate_crop() -> None:
    image = np.zeros((60, 100, 3), dtype=np.uint8)
    image[20:30, 30:50] = 255
    vehicle = _detection((10, 10, 80, 50), 0.9, 2, "car")
    local_plate = _detection((20, 10, 40, 20), 0.8)
    detector = TwoStageDetector(StubDetector([[vehicle]]), StubDetector([[local_plate]]), vehicle_padding=0)

    crops = detector.crop_plates(image)

    assert len(crops) == 1
    assert crops[0][0].shape == (10, 20, 3)
    assert np.all(crops[0][0] == 255)


def test_builds_cascade_from_config(tmp_path: Path) -> None:
    config = {
        "vehicle": {"weights": "yolo11n.pt", "class_ids": [2, 3, 5, 7]},
        "plate": {"weights": "models/plate/best.pt"},
        "cascade": {"vehicle_padding": 0.1, "plate_nms_iou": 0.4},
    }

    detector = build_two_stage_detector(config, tmp_path)

    assert detector.vehicle_detector.weights_path == Path("yolo11n.pt")
    assert detector.vehicle_detector.class_ids == COCO_VEHICLE_CLASS_IDS
    assert detector.plate_detector.weights_path == tmp_path / "models" / "plate" / "best.pt"
    assert detector.vehicle_padding == 0.1
    assert detector.plate_nms_iou == 0.4
