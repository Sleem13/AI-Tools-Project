"""Tests for the vehicle-to-license-plate YOLO cascade."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from src.detection.inference import (
    COCO_VEHICLE_CLASS_IDS,
    CharacterResult,
    DetectionResult,
    PlatePreprocessConfig,
    TwoStageDetector,
    YOLODetector,
    build_two_stage_detector,
    order_and_decode_characters,
)


class StubDetector:
    def __init__(self, batches: list[list[DetectionResult]]) -> None:
        self.batches = iter(batches)
        self.image_shapes: list[tuple[int, ...]] = []

    def predict(self, image: np.ndarray) -> list[DetectionResult]:
        self.image_shapes.append(image.shape)
        return next(self.batches)


class StubCharacterDetector:
    def __init__(self, characters: tuple[CharacterResult, ...], text: str) -> None:
        self.characters = characters
        self.text = text
        self.image_shapes: list[tuple[int, ...]] = []

    def recognize(self, image: np.ndarray):
        self.image_shapes.append(image.shape)
        return self.characters, self.text


class ShapeAwareCharacterDetector:
    def __init__(
        self,
        portrait: tuple[tuple[CharacterResult, ...], str],
        landscape: tuple[tuple[CharacterResult, ...], str],
    ) -> None:
        self.portrait = portrait
        self.landscape = landscape
        self.image_shapes: list[tuple[int, ...]] = []

    def recognize(self, image: np.ndarray):
        self.image_shapes.append(image.shape)
        h, w = image.shape[:2]
        return self.portrait if h > w else self.landscape


class ThresholdAwareCharacterDetector:
    def __init__(
        self,
        high_threshold: tuple[tuple[CharacterResult, ...], str],
        low_threshold: tuple[tuple[CharacterResult, ...], str],
    ) -> None:
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.conf_threshold = 0.15
        self.image_shapes: list[tuple[int, ...]] = []
        self.thresholds: list[float] = []

    def recognize(self, image: np.ndarray):
        self.image_shapes.append(image.shape)
        self.thresholds.append(self.conf_threshold)
        return self.low_threshold if self.conf_threshold < 0.1 else self.high_threshold


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


def test_stage_three_reads_the_stage_two_plate_crop() -> None:
    image = np.zeros((60, 100, 3), dtype=np.uint8)
    vehicle = _detection((10, 10, 80, 50), 0.9, 2, "car")
    local_plate = _detection((20, 10, 40, 20), 0.8)
    character = CharacterResult(_detection((1, 1, 4, 8), 0.95, 14, "alif"), "ا", 0, 0)
    character_detector = StubCharacterDetector((character,), "ا")
    detector = TwoStageDetector(
        StubDetector([[vehicle]]),
        StubDetector([[local_plate]]),
        vehicle_padding=0,
        character_detector=character_detector,
    )

    result = detector.predict(image)[0]

    assert character_detector.image_shapes == [(32, 64, 3)]
    assert result.character_text == "ا"
    assert result.to_dict()["characters"][0]["glyph"] == "ا"


def test_stage_three_applies_configured_crop_enhancement_before_yolo26() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    vehicle = _detection((20, 20, 160, 80), 0.9, 2, "car")
    local_plate = _detection((20, 10, 100, 50), 0.8)
    character = CharacterResult(_detection((1, 1, 4, 8), 0.95, 14, "alif"), "ا", 0, 0)
    character_detector = StubCharacterDetector((character,), "ا")
    detector = TwoStageDetector(
        StubDetector([[vehicle]]),
        StubDetector([[local_plate]]),
        vehicle_padding=0,
        character_detector=character_detector,
        character_preprocess=PlatePreprocessConfig(upscale_factor=2, clahe=True),
    )

    detector.predict(image)

    assert character_detector.image_shapes == [(80, 160, 3)]


def test_stage_three_tries_rotated_variants_for_portrait_crops() -> None:
    image = np.zeros((140, 120, 3), dtype=np.uint8)
    vehicle = _detection((10, 10, 100, 130), 0.9, 2, "car")
    local_plate = _detection((20, 10, 50, 100), 0.8)
    weak = (CharacterResult(_detection((1, 1, 4, 8), 0.6, 14, "alif"), "ا", 0, 0),)
    strong = (
        CharacterResult(_detection((1, 1, 4, 8), 0.9, 14, "alif"), "ا", 0, 0),
        CharacterResult(_detection((6, 1, 9, 8), 0.9, 30, "seen"), "س", 0, 1),
        CharacterResult(_detection((11, 1, 14, 8), 0.9, 2, "2"), "2", 0, 2),
    )
    character_detector = ShapeAwareCharacterDetector((weak, "ا"), (strong, "اس 2"))
    detector = TwoStageDetector(
        StubDetector([[vehicle]]),
        StubDetector([[local_plate]]),
        vehicle_padding=0,
        character_detector=character_detector,
        character_preprocess=PlatePreprocessConfig(upscale_factor=1, rotation_variants=True),
    )

    result = detector.predict(image)[0]

    assert character_detector.image_shapes == [(96, 32, 3), (32, 96, 3), (32, 96, 3)]
    assert result.characters == strong
    assert result.character_text == "اس 2"
    assert result.character_preprocess["selected_variant"] == "rotated_clockwise"
    assert result.character_preprocess["tried_variants"] == 3
    assert result.to_dict()["character_preprocess"]["variants"][0]["name"] == "enhanced"


def test_stage_three_retries_low_character_threshold_when_read_is_short() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    vehicle = _detection((20, 20, 180, 80), 0.9, 2, "car")
    local_plate = _detection((20, 10, 140, 45), 0.8)
    short = (
        CharacterResult(_detection((40, 1, 48, 18), 0.85, 30, "seen"), "س", 0, 0),
        CharacterResult(_detection((30, 1, 38, 18), 0.85, 14, "alif"), "ا", 0, 1),
        CharacterResult(_detection((20, 1, 28, 18), 0.85, 4, "4"), "4", 0, 2),
        CharacterResult(_detection((10, 1, 18, 18), 0.85, 5, "5"), "5", 0, 3),
        CharacterResult(_detection((1, 1, 8, 18), 0.85, 6, "6"), "6", 0, 4),
    )
    full = (
        *short,
        CharacterResult(_detection((50, 1, 58, 18), 0.06, 7, "7"), "7", 0, 5),
    )
    character_detector = ThresholdAwareCharacterDetector((short, "سا 456"), (full, "سا 7456"))
    detector = TwoStageDetector(
        StubDetector([[vehicle]]),
        StubDetector([[local_plate]]),
        vehicle_padding=0,
        character_detector=character_detector,
        character_preprocess=PlatePreprocessConfig(
            upscale_factor=1,
            retry_min_characters=6,
            retry_conf_threshold=0.05,
        ),
    )

    result = detector.predict(image)[0]

    assert character_detector.thresholds == [0.15, 0.05]
    assert character_detector.conf_threshold == 0.15
    assert result.characters == full
    assert result.character_text == "سا 7456"
    assert result.character_preprocess["selected_variant"] == "enhanced_low_conf"
    assert result.character_preprocess["variants"][1]["retry"] is True


def test_orders_egyptian_characters_right_to_left_and_separates_digits() -> None:
    detections = [
        _detection((10, 2, 18, 14), 0.9, 2, "2"),
        _detection((50, 2, 58, 14), 0.9, 14, "alif"),
        _detection((30, 2, 38, 14), 0.9, 30, "seen"),
        _detection((2, 2, 8, 14), 0.9, 1, "1"),
    ]

    ordered, text = order_and_decode_characters(detections)

    assert [item.glyph for item in ordered] == ["ا", "س", "2", "1"]
    assert text == "اس 21"


def test_egyptian_decoder_places_letters_before_numbers_even_when_geometry_is_noisy() -> None:
    detections = [
        _detection((70, 2, 78, 14), 0.9, 2, "2"),
        _detection((50, 2, 58, 14), 0.9, 1, "1"),
        _detection((30, 2, 38, 14), 0.9, 30, "seen"),
        _detection((10, 2, 18, 14), 0.9, 14, "alif"),
    ]

    ordered, text = order_and_decode_characters(detections)

    assert [item.glyph for item in ordered] == ["س", "ا", "2", "1"]
    assert text == "سا 21"


def test_character_decoder_preserves_separate_rows() -> None:
    detections = [
        _detection((30, 2, 38, 12), 0.9, 14, "alif"),
        _detection((10, 2, 18, 12), 0.9, 30, "seen"),
        _detection((30, 30, 38, 40), 0.9, 1, "1"),
        _detection((10, 30, 18, 40), 0.9, 2, "2"),
    ]

    ordered, text = order_and_decode_characters(detections)

    assert [item.row for item in ordered] == [0, 0, 1, 1]
    assert text == "اس / 12"


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


def test_bare_configured_weights_use_project_file_when_present(tmp_path: Path) -> None:
    (tmp_path / "yolo11n.pt").touch()

    config = {
        "vehicle": {"weights": "yolo11n.pt"},
        "plate": {"weights": "plate.pt"},
    }

    detector = build_two_stage_detector(config, tmp_path)

    assert detector.vehicle_detector.weights_path == tmp_path / "yolo11n.pt"


def test_builds_character_stage_when_checkpoint_exists(tmp_path: Path) -> None:
    character_weights = tmp_path / "models" / "character" / "best.pt"
    character_weights.parent.mkdir(parents=True)
    character_weights.touch()
    config = {
        "vehicle": {"weights": "yolo11n.pt"},
        "plate": {"weights": "plate.pt"},
        "character": {
            "enabled": True,
            "weights": "models/character/best.pt",
            "reading_direction": "rtl",
            "preprocess": {
                "upscale_factor": 2,
                "clahe": True,
                "sharpen": True,
                "sharpen_amount": 0.2,
                "rotation_variants": True,
                "rotation_variant_min_aspect": 1.1,
            },
        },
    }

    detector = build_two_stage_detector(config, tmp_path)

    assert detector.character_detector is not None
    assert detector.character_detector.weights_path == character_weights
    assert detector.character_preprocess.upscale_factor == 2
    assert detector.character_preprocess.clahe is True
    assert detector.character_preprocess.sharpen is True
    assert detector.character_preprocess.sharpen_amount == pytest.approx(0.2)
    assert detector.character_preprocess.rotation_variants is True
    assert detector.character_preprocess.rotation_variant_min_aspect == pytest.approx(1.1)
    assert detector.character_preprocess.retry_min_characters == 6
    assert detector.character_preprocess.retry_conf_threshold == pytest.approx(0.05)
