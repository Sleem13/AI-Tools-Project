"""YOLO inference primitives for vehicle, plate, and cascaded detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

COCO_VEHICLE_CLASS_IDS = frozenset({2, 3, 5, 7})

CHARACTER_GLYPHS = {
    "7aa": "ح",
    "Taa": "ط",
    "Thaa": "ظ",
    "ain": "ع",
    "alif": "ا",
    "baa": "ب",
    "daad": "ض",
    "daal": "د",
    "faa": "ف",
    "ghayn": "غ",
    "haa": "ه",
    "jeem": "ج",
    "kaaf": "ك",
    "khaa": "خ",
    "laam": "ل",
    "meem": "م",
    "noon": "ن",
    "qaaf": "ق",
    "raa": "ر",
    "saad": "ص",
    "seen": "س",
    "sheen": "ش",
    "taa": "ت",
    "thaa": "ث",
    "waw": "و",
    "yaa": "ي",
    "zaal": "ذ",
    "zay": "ز",
}


@dataclass(frozen=True)
class DetectionResult:
    """One object detection in absolute ``xyxy`` pixel coordinates."""

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


@dataclass(frozen=True)
class CharacterResult:
    """One ordered character detected inside a stage-two plate crop."""

    detection: DetectionResult
    glyph: str
    row: int
    order: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": list(self.detection.bbox),
            "confidence": self.detection.confidence,
            "class_id": self.detection.class_id,
            "class_name": self.detection.class_name,
            "glyph": self.glyph,
            "row": self.row,
            "order": self.order,
        }


@dataclass(frozen=True)
class TwoStageDetection:
    """A plate detection associated with the vehicle that produced it."""

    vehicle: DetectionResult
    plate: DetectionResult
    plate_bbox_in_vehicle: tuple[float, float, float, float]
    characters: tuple[CharacterResult, ...] = ()
    character_text: str = ""
    character_preprocess: dict[str, Any] = field(default_factory=dict)

    @property
    def combined_confidence(self) -> float:
        """Conservative cascade score used for ranking duplicate plates."""
        return self.vehicle.confidence * self.plate.confidence

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "bbox": list(self.plate.bbox),
            "confidence": self.plate.confidence,
            "combined_confidence": self.combined_confidence,
            "class_id": self.plate.class_id,
            "class_name": self.plate.class_name,
            "vehicle": {
                "bbox": list(self.vehicle.bbox),
                "confidence": self.vehicle.confidence,
                "class_id": self.vehicle.class_id,
                "class_name": self.vehicle.class_name,
            },
        }
        if self.characters:
            payload["characters"] = [character.to_dict() for character in self.characters]
            payload["character_text"] = self.character_text
        if self.character_preprocess:
            payload["character_preprocess"] = self.character_preprocess
        return payload


@dataclass(frozen=True)
class PlatePreprocessConfig:
    """Model-facing crop preparation before stage-three character detection."""

    enabled: bool = True
    upscale_factor: int = 1
    min_dim: int = 32
    target_height: int = 128
    clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    sharpen: bool = False
    sharpen_amount: float = 0.4
    rotation_variants: bool = True
    rotation_variant_min_aspect: float = 1.15
    retry_min_characters: int = 6
    retry_conf_threshold: float = 0.05

    def __post_init__(self) -> None:
        if not 1 <= self.upscale_factor <= 8:
            raise ValueError("upscale_factor must be between 1 and 8")
        if self.min_dim <= 0:
            raise ValueError("min_dim must be positive")
        if self.target_height <= 0:
            raise ValueError("target_height must be positive")
        if self.clahe_clip_limit <= 0:
            raise ValueError("clahe_clip_limit must be positive")
        if self.clahe_tile_grid_size <= 0:
            raise ValueError("clahe_tile_grid_size must be positive")
        if self.sharpen_amount < 0:
            raise ValueError("sharpen_amount must be non-negative")
        if self.rotation_variant_min_aspect <= 0:
            raise ValueError("rotation_variant_min_aspect must be positive")
        if self.retry_min_characters < 0:
            raise ValueError("retry_min_characters must be non-negative")
        if not 0 < self.retry_conf_threshold <= 1:
            raise ValueError("retry_conf_threshold must be between 0 and 1")


class YOLODetector:
    """Lazy Ultralytics detector with consistent filtering and result parsing."""

    def __init__(
        self,
        weights_path: str | Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str | int = "cpu",
        class_ids: set[int] | frozenset[int] | None = None,
        max_detections: int = 300,
        agnostic_nms: bool = False,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.class_ids = frozenset(class_ids) if class_ids is not None else None
        self.max_detections = max_detections
        self.agnostic_nms = agnostic_nms
        self._model = None

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self.weights_path))
        return self._model

    def predict(self, image: np.ndarray) -> list[DetectionResult]:
        """Run detection on one BGR image."""
        if not isinstance(image, np.ndarray) or image.ndim not in {2, 3} or image.size == 0:
            raise ValueError("image must be a non-empty grayscale or BGR numpy array")

        h, w = image.shape[:2]
        if h < 16 or w < 16:
            return []

        model = self._load_model()
        predict_args: dict[str, Any] = {
            "source": image,
            "conf": self.conf_threshold,
            "iou": self.iou_threshold,
            "device": self.device,
            "max_det": self.max_detections,
            "agnostic_nms": self.agnostic_nms,
            "verbose": False,
        }
        if self.class_ids is not None:
            predict_args["classes"] = sorted(self.class_ids)
        results = model.predict(**predict_args)
        return self._parse_results(results)

    def predict_path(self, image_path: str | Path) -> list[DetectionResult]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return self.predict(image)

    def _parse_results(self, results: Any) -> list[DetectionResult]:
        detections: list[DetectionResult] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0])
                if self.class_ids is not None and class_id not in self.class_ids:
                    continue
                detections.append(
                    DetectionResult(
                        bbox=tuple(float(value) for value in box.xyxy[0].tolist()),
                        confidence=float(box.conf[0]),
                        class_id=class_id,
                        class_name=_class_name(result.names, class_id),
                    )
                )
        return detections


class VehicleDetector(YOLODetector):
    """YOLO11 detector restricted to road-vehicle classes from COCO."""

    def __init__(
        self,
        weights_path: str | Path = "yolo11n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str | int = "cpu",
        class_ids: set[int] | frozenset[int] = COCO_VEHICLE_CLASS_IDS,
        max_detections: int = 100,
    ) -> None:
        super().__init__(
            weights_path=weights_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device,
            class_ids=class_ids,
            max_detections=max_detections,
        )


class LicensePlateDetector(YOLODetector):
    """Custom YOLO11 plate detector, usable alone or as cascade stage two."""

    def crop_plates(self, image: np.ndarray, min_confidence: float = 0.25) -> list[tuple[np.ndarray, float]]:
        detections = self.predict(image)
        crops: list[tuple[np.ndarray, float]] = []
        for detection in detections:
            if detection.confidence < min_confidence:
                continue
            clipped = _clip_bbox(detection.bbox, image.shape[1], image.shape[0])
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            crops.append((image[y1:y2, x1:x2].copy(), detection.confidence))
        return crops


class CharacterDetector(YOLODetector):
    """YOLO26 character detector plus deterministic Egyptian plate decoding."""

    def __init__(
        self,
        *args,
        reading_direction: str = "rtl",
        row_threshold: float = 0.6,
        agnostic_nms: bool = False,
        **kwargs,
    ) -> None:
        kwargs.setdefault("agnostic_nms", agnostic_nms)
        super().__init__(*args, **kwargs)
        if reading_direction not in {"ltr", "rtl"}:
            raise ValueError("reading_direction must be 'ltr' or 'rtl'")
        if row_threshold <= 0:
            raise ValueError("row_threshold must be positive")
        self.reading_direction = reading_direction
        self.row_threshold = row_threshold

    def recognize(self, plate_crop: np.ndarray) -> tuple[tuple[CharacterResult, ...], str]:
        detections = self.predict(plate_crop)
        detections = _deduplicate_characters(detections, plate_crop.shape)
        return order_and_decode_characters(
            detections,
            reading_direction=self.reading_direction,
            row_threshold=self.row_threshold,
        )


class TwoStageDetector:
    """Detect vehicles, then detect plates only inside each vehicle crop."""

    def __init__(
        self,
        vehicle_detector: YOLODetector,
        plate_detector: YOLODetector,
        vehicle_padding: float = 0.05,
        plate_nms_iou: float = 0.5,
        fallback_to_full_image: bool = False,
        character_detector: CharacterDetector | None = None,
        character_preprocess: PlatePreprocessConfig | None = None,
    ) -> None:
        if vehicle_padding < 0:
            raise ValueError("vehicle_padding must be non-negative")
        if not 0 <= plate_nms_iou <= 1:
            raise ValueError("plate_nms_iou must be between 0 and 1")
        self.vehicle_detector = vehicle_detector
        self.plate_detector = plate_detector
        self.vehicle_padding = vehicle_padding
        self.plate_nms_iou = plate_nms_iou
        self.fallback_to_full_image = fallback_to_full_image
        self.character_detector = character_detector
        self.character_preprocess = character_preprocess or PlatePreprocessConfig()

    def predict(self, image: np.ndarray) -> list[TwoStageDetection]:
        if not isinstance(image, np.ndarray) or image.ndim not in {2, 3} or image.size == 0:
            raise ValueError("image must be a non-empty grayscale or BGR numpy array")

        height, width = image.shape[:2]
        vehicles = self.vehicle_detector.predict(image)
        if not vehicles and self.fallback_to_full_image:
            vehicles = [
                DetectionResult(
                    bbox=(0.0, 0.0, float(width), float(height)),
                    confidence=1.0,
                    class_id=-1,
                    class_name="full_image",
                )
            ]

        candidates: list[TwoStageDetection] = []
        for vehicle in vehicles:
            crop_bbox = _expanded_bbox(vehicle.bbox, width, height, self.vehicle_padding)
            if crop_bbox is None:
                continue
            crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
            vehicle_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
            for local_plate in self.plate_detector.predict(vehicle_crop):
                local_clipped = _clip_bbox(local_plate.bbox, crop_x2 - crop_x1, crop_y2 - crop_y1)
                if local_clipped is None:
                    continue
                local_x1, local_y1, local_x2, local_y2 = local_clipped
                global_bbox = (
                    float(crop_x1 + local_x1),
                    float(crop_y1 + local_y1),
                    float(crop_x1 + local_x2),
                    float(crop_y1 + local_y2),
                )
                characters: tuple[CharacterResult, ...] = ()
                character_text = ""
                character_preprocess: dict[str, Any] = {}
                if self.character_detector is not None:
                    plate_crop = vehicle_crop[local_y1:local_y2, local_x1:local_x2]
                    plate_crop = preprocess_plate(plate_crop, self.character_preprocess)
                    if plate_crop is not None:
                        characters, character_text, character_preprocess = _recognize_character_variants(
                            self.character_detector,
                            plate_crop,
                            self.character_preprocess,
                        )
                candidates.append(
                    TwoStageDetection(
                        vehicle=vehicle,
                        plate=DetectionResult(
                            bbox=global_bbox,
                            confidence=local_plate.confidence,
                            class_id=local_plate.class_id,
                            class_name=local_plate.class_name,
                        ),
                        plate_bbox_in_vehicle=tuple(float(value) for value in local_clipped),
                        characters=characters,
                        character_text=character_text,
                        character_preprocess=character_preprocess,
                    )
                )
        return _deduplicate_plates(candidates, self.plate_nms_iou)

    def predict_path(self, image_path: str | Path) -> list[TwoStageDetection]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return self.predict(image)

    def crop_plates(self, image: np.ndarray) -> list[tuple[np.ndarray, TwoStageDetection]]:
        crops: list[tuple[np.ndarray, TwoStageDetection]] = []
        for detection in self.predict(image):
            clipped = _clip_bbox(detection.plate.bbox, image.shape[1], image.shape[0])
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            crops.append((image[y1:y2, x1:x2].copy(), detection))
        return crops


def build_two_stage_detector(
    config: dict[str, Any],
    project_root: str | Path | None = None,
) -> TwoStageDetector:
    """Construct the cascade from ``configs/model/two_stage.yaml`` data."""
    root = Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
    vehicle_config = config.get("vehicle", {})
    plate_config = config.get("plate", {})
    character_config = config.get("character", {})
    cascade_config = config.get("cascade", {})
    vehicle_detector = VehicleDetector(
        weights_path=_configured_weights(vehicle_config.get("weights", "yolo11n.pt"), root),
        conf_threshold=vehicle_config.get("conf_threshold", 0.3),
        iou_threshold=vehicle_config.get("iou_threshold", 0.45),
        device=vehicle_config.get("device", "cpu"),
        class_ids=set(vehicle_config.get("class_ids", COCO_VEHICLE_CLASS_IDS)),
        max_detections=vehicle_config.get("max_detections", 100),
    )
    plate_detector = LicensePlateDetector(
        weights_path=_configured_weights(plate_config["weights"], root),
        conf_threshold=plate_config.get("conf_threshold", 0.25),
        iou_threshold=plate_config.get("iou_threshold", 0.45),
        device=plate_config.get("device", vehicle_config.get("device", "cpu")),
        max_detections=plate_config.get("max_detections", 20),
    )
    character_detector = None
    if character_config.get("enabled", False):
        character_weights = _configured_weights(character_config["weights"], root)
        if _weights_exist(character_weights):
            character_detector = CharacterDetector(
                weights_path=character_weights,
                conf_threshold=character_config.get("conf_threshold", 0.15),
                iou_threshold=character_config.get("iou_threshold", 0.45),
                device=character_config.get("device", plate_config.get("device", vehicle_config.get("device", "cpu"))),
                max_detections=character_config.get("max_detections", 12),
                reading_direction=character_config.get("reading_direction", "rtl"),
                row_threshold=character_config.get("row_threshold", 0.6),
                agnostic_nms=character_config.get("agnostic_nms", False),
            )
    return TwoStageDetector(
        vehicle_detector=vehicle_detector,
        plate_detector=plate_detector,
        vehicle_padding=cascade_config.get("vehicle_padding", 0.05),
        plate_nms_iou=cascade_config.get("plate_nms_iou", 0.5),
        fallback_to_full_image=cascade_config.get("fallback_to_full_image", False),
        character_detector=character_detector,
        character_preprocess=_build_plate_preprocess_config(character_config.get("preprocess", {})),
    )


def order_and_decode_characters(
    detections: list[DetectionResult],
    reading_direction: str = "rtl",
    row_threshold: float = 0.6,
) -> tuple[tuple[CharacterResult, ...], str]:
    """Cluster characters into rows and decode each row in reading order."""
    if not detections:
        return (), ""
    if reading_direction not in {"ltr", "rtl"}:
        raise ValueError("reading_direction must be 'ltr' or 'rtl'")

    median_height = float(np.median([item.bbox[3] - item.bbox[1] for item in detections]))
    tolerance = max(1.0, median_height * row_threshold)
    rows: list[list[DetectionResult]] = []
    row_centers: list[float] = []
    for detection in sorted(detections, key=lambda item: _bbox_center(item.bbox)[1]):
        center_y = _bbox_center(detection.bbox)[1]
        if not rows:
            rows.append([detection])
            row_centers.append(center_y)
            continue
        closest = min(range(len(rows)), key=lambda index: abs(center_y - row_centers[index]))
        if abs(center_y - row_centers[closest]) <= tolerance:
            rows[closest].append(detection)
            row_centers[closest] = sum(_bbox_center(item.bbox)[1] for item in rows[closest]) / len(rows[closest])
        else:
            rows.append([detection])
            row_centers.append(center_y)

    ordered_rows = sorted(zip(row_centers, rows, strict=True), key=lambda item: item[0])
    results: list[CharacterResult] = []
    row_texts: list[str] = []
    for row_index, (_, row) in enumerate(ordered_rows):
        ordered = _order_plate_row(row, reading_direction)
        glyphs = []
        previous_kind = None
        for order, detection in enumerate(ordered):
            glyph = CHARACTER_GLYPHS.get(detection.class_name, detection.class_name)
            kind = "digit" if glyph.isdigit() else "letter"
            if glyphs and kind != previous_kind:
                glyphs.append(" ")
            glyphs.append(glyph)
            previous_kind = kind
            results.append(
                CharacterResult(
                    detection=detection,
                    glyph=glyph,
                    row=row_index,
                    order=order,
                )
            )
        row_texts.append("".join(glyphs))
    return tuple(results), " / ".join(row_texts)


def _order_plate_row(detections: list[DetectionResult], reading_direction: str) -> list[DetectionResult]:
    if reading_direction == "ltr":
        return sorted(detections, key=lambda item: _bbox_center(item.bbox)[0])

    letters: list[DetectionResult] = []
    digits: list[DetectionResult] = []
    others: list[DetectionResult] = []
    for detection in detections:
        glyph = CHARACTER_GLYPHS.get(detection.class_name, detection.class_name)
        if glyph.isdigit():
            digits.append(detection)
        elif glyph:
            letters.append(detection)
        else:
            others.append(detection)

    return [
        *sorted(letters, key=lambda item: _bbox_center(item.bbox)[0], reverse=True),
        *sorted(digits, key=lambda item: _bbox_center(item.bbox)[0], reverse=True),
        *sorted(others, key=lambda item: _bbox_center(item.bbox)[0], reverse=True),
    ]


def _class_name(names: dict[int, str] | list[str], class_id: int) -> str:
    if isinstance(names, dict):
        return names.get(class_id, str(class_id))
    if 0 <= class_id < len(names):
        return names[class_id]
    return str(class_id)


def _configured_weights(value: str | Path, project_root: Path) -> str | Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parent == Path("."):
        project_path = project_root / path
        if project_path.is_file():
            return project_path
        return str(value)
    return project_root / path


def _build_plate_preprocess_config(config: dict[str, Any] | None) -> PlatePreprocessConfig:
    if not config:
        return PlatePreprocessConfig()
    return PlatePreprocessConfig(
        enabled=config.get("enabled", True),
        upscale_factor=config.get("upscale_factor", 1),
        min_dim=config.get("min_dim", 32),
        target_height=config.get("target_height", 128),
        clahe=config.get("clahe", False),
        clahe_clip_limit=config.get("clahe_clip_limit", 2.0),
        clahe_tile_grid_size=config.get("clahe_tile_grid_size", 8),
        sharpen=config.get("sharpen", False),
        sharpen_amount=config.get("sharpen_amount", 0.4),
        rotation_variants=config.get("rotation_variants", True),
        rotation_variant_min_aspect=config.get("rotation_variant_min_aspect", 1.15),
        retry_min_characters=config.get("retry_min_characters", 6),
        retry_conf_threshold=config.get("retry_conf_threshold", 0.05),
    )


def _recognize_character_variants(
    character_detector: CharacterDetector,
    plate_crop: np.ndarray,
    config: PlatePreprocessConfig,
) -> tuple[tuple[CharacterResult, ...], str, dict[str, Any]]:
    best_characters: tuple[CharacterResult, ...] = ()
    best_text = ""
    best_name = ""
    best_score = (-1, -1.0)
    variant_reports: list[dict[str, Any]] = []
    for name, candidate in _character_crop_variants(plate_crop, config):
        attempts = _recognize_character_attempts(character_detector, candidate, name, config)
        for attempt in attempts:
            variant_reports.append(attempt)
            score = (attempt["characters"], attempt["average_confidence"])
            if score > best_score:
                best_characters = attempt["ordered_characters"]
                best_text = attempt["text"]
                best_name = attempt["name"]
                best_score = score
    for report in variant_reports:
        report.pop("ordered_characters", None)
    return best_characters, best_text, {
        "selected_variant": best_name,
        "tried_variants": len(variant_reports),
        "variants": variant_reports,
    }


def _recognize_character_attempts(
    character_detector: CharacterDetector,
    plate_crop: np.ndarray,
    name: str,
    config: PlatePreprocessConfig,
) -> list[dict[str, Any]]:
    original_threshold = getattr(character_detector, "conf_threshold", None)
    attempts: list[dict[str, Any]] = []
    characters, text = character_detector.recognize(plate_crop)
    attempts.append(
        _character_attempt_report(
            name=name,
            plate_crop=plate_crop,
            characters=characters,
            text=text,
            threshold=original_threshold,
            retry=False,
        )
    )

    if (
        config.retry_min_characters
        and len(characters) < config.retry_min_characters
        and isinstance(original_threshold, (int, float))
        and config.retry_conf_threshold < float(original_threshold)
    ):
        try:
            character_detector.conf_threshold = config.retry_conf_threshold
            retry_characters, retry_text = character_detector.recognize(plate_crop)
        finally:
            character_detector.conf_threshold = original_threshold
        attempts.append(
            _character_attempt_report(
                name=f"{name}_low_conf",
                plate_crop=plate_crop,
                characters=retry_characters,
                text=retry_text,
                threshold=config.retry_conf_threshold,
                retry=True,
            )
        )
    return attempts


def _character_attempt_report(
    name: str,
    plate_crop: np.ndarray,
    characters: tuple[CharacterResult, ...],
    text: str,
    threshold: float | None,
    retry: bool,
) -> dict[str, Any]:
    return {
        "name": name,
        "shape": list(plate_crop.shape),
        "characters": len(characters),
        "average_confidence": _average_character_confidence(characters),
        "threshold": threshold,
        "retry": retry,
        "text": text,
        "ordered_characters": characters,
    }


def _character_crop_variants(plate_crop: np.ndarray, config: PlatePreprocessConfig) -> list[tuple[str, np.ndarray]]:
    variants = [("enhanced", plate_crop)]
    h, w = plate_crop.shape[:2]
    if config.rotation_variants and h > w * config.rotation_variant_min_aspect:
        variants.append(("rotated_clockwise", cv2.rotate(plate_crop, cv2.ROTATE_90_CLOCKWISE)))
        variants.append(("rotated_counterclockwise", cv2.rotate(plate_crop, cv2.ROTATE_90_COUNTERCLOCKWISE)))
    return variants


def _average_character_confidence(characters: tuple[CharacterResult, ...]) -> float:
    if not characters:
        return 0.0
    return sum(character.detection.confidence for character in characters) / len(characters)


def _weights_exist(value: str | Path) -> bool:
    path = Path(value)
    return path.is_file()


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _clip_bbox(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    clipped = (
        max(0, min(image_width, int(np.floor(x1)))),
        max(0, min(image_height, int(np.floor(y1)))),
        max(0, min(image_width, int(np.ceil(x2)))),
        max(0, min(image_height, int(np.ceil(y2)))),
    )
    return clipped if clipped[2] > clipped[0] and clipped[3] > clipped[1] else None


def _expanded_bbox(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    padding: float,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    pad_x = max(0.0, x2 - x1) * padding
    pad_y = max(0.0, y2 - y1) * padding
    return _clip_bbox((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), image_width, image_height)


def _bbox_iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _deduplicate_plates(
    detections: list[TwoStageDetection],
    iou_threshold: float,
) -> list[TwoStageDetection]:
    kept: list[TwoStageDetection] = []
    for candidate in sorted(detections, key=lambda item: item.combined_confidence, reverse=True):
        if all(_bbox_iou(candidate.plate.bbox, existing.plate.bbox) <= iou_threshold for existing in kept):
            kept.append(candidate)
    return kept


PLATE_TARGET_HEIGHT = 128
PLATE_MIN_DIM = 32


def preprocess_plate(
    plate_crop: np.ndarray,
    config: PlatePreprocessConfig | None = None,
) -> np.ndarray | None:
    """Prepare plate crops before character detection.

    Upscales/enhances the stage-two crop, then tries perspective-warping
    the plate area. Falls back to the enhanced crop if contour detection fails.
    """
    if plate_crop.size == 0:
        return None

    config = config or PlatePreprocessConfig()
    if not config.enabled:
        return plate_crop

    plate_crop = _enhance_character_crop(plate_crop, config)
    h, w = plate_crop.shape[:2]

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if plate_crop.ndim == 3 else plate_crop
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blurred, 30, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return plate_crop

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    best_quad = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.3 * h * w:
            best_quad = approx.reshape(4, 2).astype(np.float32)
            break

    if best_quad is None:
        return plate_crop

    corners = _order_plate_corners(best_quad)
    aspect = w / max(h, 1)
    target_h = config.target_height
    target_w = max(config.min_dim, int(target_h * aspect))
    if target_w > target_h * 4:
        target_w = target_h * 4
    dst = np.array([[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(plate_crop, matrix, (target_w, target_h))
    if warped.shape[1] < config.min_dim or warped.shape[0] < config.min_dim:
        return None
    return warped


def _enhance_character_crop(plate_crop: np.ndarray, config: PlatePreprocessConfig) -> np.ndarray:
    h, w = plate_crop.shape[:2]
    scale = max(float(config.upscale_factor), config.min_dim / max(1, min(h, w)))
    if scale > 1:
        plate_crop = cv2.resize(
            plate_crop,
            (max(config.min_dim, round(w * scale)), max(config.min_dim, round(h * scale))),
            interpolation=cv2.INTER_LANCZOS4,
        )

    if config.clahe:
        clahe_op = cv2.createCLAHE(
            clipLimit=config.clahe_clip_limit,
            tileGridSize=(config.clahe_tile_grid_size, config.clahe_tile_grid_size),
        )
        if plate_crop.ndim == 2:
            plate_crop = clahe_op.apply(plate_crop)
        else:
            lab = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe_op.apply(lab[:, :, 0])
            plate_crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if config.sharpen and config.sharpen_amount > 0:
        blurred = cv2.GaussianBlur(plate_crop, (0, 0), sigmaX=3)
        plate_crop = cv2.addWeighted(plate_crop, 1 + config.sharpen_amount, blurred, -config.sharpen_amount, 0)

    return plate_crop


def _order_plate_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def _deduplicate_characters(
    detections: list[DetectionResult],
    image_shape: tuple[int, ...],
    distance_ratio: float = 0.15,
) -> list[DetectionResult]:
    """Remove duplicate character detections whose centers are too close."""
    if len(detections) <= 1:
        return list(detections)
    h = float(image_shape[0])
    threshold = max(4.0, h * distance_ratio)
    kept: list[DetectionResult] = []
    for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
        cx, cy = _bbox_center(det.bbox)
        too_close = False
        for existing in kept:
            ex, ey = _bbox_center(existing.bbox)
            if abs(cx - ex) < threshold and abs(cy - ey) < threshold:
                too_close = True
                break
        if not too_close:
            kept.append(det)
    return kept
