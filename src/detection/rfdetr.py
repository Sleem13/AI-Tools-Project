"""RF-DETR dataset conversion and optional fine-tuning helpers."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
SPLIT_MAP = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}


@dataclass(frozen=True)
class RFDatasetSummary:
    output_dir: Path
    images: dict[str, int]
    annotations: dict[str, int]
    categories: dict[int, str]


def convert_yolo_dataset_to_rfdetr_coco(
    data_yaml: str | Path,
    output_dir: str | Path,
    *,
    copy_images: bool = True,
) -> RFDatasetSummary:
    """Convert an Ultralytics YOLO dataset YAML into RF-DETR's COCO folder layout."""
    data_yaml = Path(data_yaml).resolve()
    output_dir = Path(output_dir).resolve()
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    dataset_root = _dataset_root(config, data_yaml)
    categories = _normalise_names(config.get("names", {0: "license_plate"}))
    image_counts: dict[str, int] = {}
    annotation_counts: dict[str, int] = {}

    for yolo_split, rf_split in SPLIT_MAP.items():
        if yolo_split not in config:
            continue
        images = _split_images(config[yolo_split], dataset_root)
        split_dir = output_dir / rf_split
        split_dir.mkdir(parents=True, exist_ok=True)
        coco = {
            "images": [],
            "annotations": [],
            "categories": [{"id": class_id + 1, "name": name} for class_id, name in sorted(categories.items())],
        }
        annotation_id = 1
        for image_id, image_path in enumerate(images, start=1):
            width, height = _image_size(image_path)
            target_name = image_path.name
            if copy_images:
                shutil.copy2(image_path, split_dir / target_name)
            else:
                target_name = str(image_path)
            coco["images"].append(
                {
                    "id": image_id,
                    "file_name": target_name,
                    "width": width,
                    "height": height,
                }
            )
            label_path = _label_path_for_image(image_path)
            for class_id, bbox in _read_yolo_labels(label_path, width, height):
                if class_id not in categories:
                    continue
                x, y, w, h = bbox
                coco["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id + 1,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1
        (split_dir / "_annotations.coco.json").write_text(
            json.dumps(coco, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        image_counts[rf_split] = len(coco["images"])
        annotation_counts[rf_split] = len(coco["annotations"])

    return RFDatasetSummary(
        output_dir=output_dir,
        images=image_counts,
        annotations=annotation_counts,
        categories=categories,
    )


def train_rfdetr(config: dict[str, Any], dataset_dir: str | Path):
    """Train RF-DETR from an RF-DETR/COCO dataset directory."""
    try:
        from rfdetr import RFDETRBase
    except ImportError as exc:
        raise RuntimeError(
            "RF-DETR is not installed in this environment. Install it with `pip install rfdetr` "
            "or use a dedicated GPU environment before training."
        ) from exc

    model_config = config.get("model", {})
    training = config.get("training", {})
    model_kwargs = {}
    if model_config.get("pretrain_weights"):
        model_kwargs["pretrain_weights"] = model_config["pretrain_weights"]
    model = RFDETRBase(**model_kwargs)
    train_kwargs = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(training.get("output_dir", "models/rfdetr/plate")),
        "epochs": training.get("epochs", 25),
        "batch_size": training.get("batch_size", 4),
        "grad_accum_steps": training.get("grad_accum_steps", 4),
        "lr": training.get("lr", 1e-4),
        "lr_encoder": training.get("lr_encoder"),
        "resolution": training.get("resolution", 672),
        "weight_decay": training.get("weight_decay", 1e-4),
        "device": training.get("device", "cuda"),
        "use_ema": training.get("use_ema", True),
        "gradient_checkpointing": training.get("gradient_checkpointing", True),
        "checkpoint_interval": training.get("checkpoint_interval", 5),
        "tensorboard": training.get("tensorboard", False),
        "wandb": training.get("wandb", False),
        "early_stopping": training.get("early_stopping", True),
        "early_stopping_patience": training.get("early_stopping_patience", 10),
        "early_stopping_min_delta": training.get("early_stopping_min_delta", 0.001),
    }
    return model.train(**{key: value for key, value in train_kwargs.items() if value is not None})


def _dataset_root(config: dict[str, Any], data_yaml: Path) -> Path:
    root = Path(config.get("path", data_yaml.parent))
    return root if root.is_absolute() else (data_yaml.parent / root).resolve()


def _normalise_names(names: dict[Any, str] | list[str]) -> dict[int, str]:
    if isinstance(names, list):
        return dict(enumerate(names))
    return {int(class_id): str(name) for class_id, name in names.items()}


def _split_images(split_value: str | list[str], dataset_root: Path) -> list[Path]:
    values = split_value if isinstance(split_value, list) else [split_value]
    images: list[Path] = []
    for value in values:
        path = Path(value)
        path = path if path.is_absolute() else dataset_root / path
        if path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                image_path = Path(line)
                images.append(image_path if image_path.is_absolute() else (path.parent / image_path).resolve())
        elif path.is_dir():
            images.extend(sorted(item for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES))
        elif any(char in str(path) for char in "*?[]"):
            images.extend(sorted(item for item in path.parent.glob(path.name) if item.suffix.lower() in IMAGE_SUFFIXES))
        else:
            raise FileNotFoundError(f"Split path does not exist: {path}")
    return sorted(dict.fromkeys(image.resolve() for image in images))


def _image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def _label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def _read_yolo_labels(label_path: Path, image_width: int, image_height: int) -> list[tuple[int, tuple[float, float, float, float]]]:
    if not label_path.is_file():
        return []
    rows: list[tuple[int, tuple[float, float, float, float]]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        cx, cy, width, height = (float(value) for value in parts[1:5])
        box_w = width * image_width
        box_h = height * image_height
        x = (cx * image_width) - (box_w / 2)
        y = (cy * image_height) - (box_h / 2)
        rows.append((class_id, (max(0.0, x), max(0.0, y), box_w, box_h)))
    return rows
