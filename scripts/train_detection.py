#!/usr/bin/env python3
"""Train a YOLO vehicle, plate, or character detection stage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.data import create_dataset_yaml
from src.detection.trainer import train_detector

DEFAULT_CONFIGS = {
    "plate": PROJECT_ROOT / "configs" / "model" / "master_plate_detection.yaml",
    "vehicle": PROJECT_ROOT / "configs" / "model" / "vehicle_detection.yaml",
    "character": PROJECT_ROOT / "configs" / "model" / "character_detection.yaml",
}

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _count_split_images(split_path: str | Path) -> tuple[int, list[Path]]:
    """Return the usable image count and missing manifest entries for one split."""
    path = Path(split_path)
    if path.is_dir():
        return sum(
            1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        ), []
    if path.is_file() and path.suffix.lower() == ".txt":
        entries = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        resolved = [
            Path(entry) if Path(entry).is_absolute() else (path.parent / entry).resolve()
            for entry in entries
        ]
        missing = [entry for entry in resolved if not entry.is_file()]
        return len(resolved) - len(missing), missing
    return 0, [path]


def validate_training_setup(dataset_yaml: Path, device: str) -> dict[str, Any]:
    """Fail fast on missing data or an incompatible CUDA selection."""
    if not dataset_yaml.is_file():
        raise RuntimeError(
            f"Dataset YAML not found: {dataset_yaml}\n"
            "Prepare the dataset first or pass its YAML with --data."
        )

    from ultralytics.data.utils import check_det_dataset

    try:
        dataset = check_det_dataset(str(dataset_yaml), autodownload=False)
    except Exception as exc:
        raise RuntimeError(f"Invalid detection dataset {dataset_yaml}: {exc}") from exc

    split_counts: dict[str, int] = {}
    for split in ("train", "val"):
        split_path = dataset.get(split)
        if not split_path:
            raise RuntimeError(f"Dataset {dataset_yaml} does not define a {split} split.")
        count, missing = _count_split_images(split_path)
        if missing:
            example = missing[0]
            raise RuntimeError(
                f"{split} split references {len(missing)} missing file(s); first missing: {example}\n"
                "Restore/regenerate the dataset and split manifests before training."
            )
        if count == 0:
            raise RuntimeError(f"{split} split contains no supported images: {split_path}")
        split_counts[split] = count

    normalized_device = str(device).strip().lower()
    if normalized_device not in {"cpu", "mps"}:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Device {device!r} requests CUDA, but this Python environment has "
                f"PyTorch {torch.__version__} without available CUDA.\n"
                "Use .venv-gpu\\Scripts\\python.exe after installing requirements-gpu.txt, "
                "or pass --device cpu for a slow smoke run."
            )

    return {
        "classes": len(dataset["names"]),
        "train_images": split_counts["train"],
        "val_images": split_counts["val"],
        "device": device,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO stage for the vehicle-to-plate-to-character cascade.")
    parser.add_argument("--stage", choices=("vehicle", "plate", "character"), default="plate")
    parser.add_argument("--config", type=Path, default=None, help="Override the stage configuration")
    parser.add_argument("--data", type=Path, default=None, help="Use an existing Ultralytics dataset YAML")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="GPU id such as 0, or cpu")
    parser.add_argument("--workers", type=int, default=None, help="Override data-loader workers")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--name", type=str, default=None, help="Override the Ultralytics run name")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the dataset and selected device without downloading a model or training",
    )
    args = parser.parse_args()

    config_path = args.config or DEFAULT_CONFIGS[args.stage]
    config = load_config(config_path)
    hyperparameters = config.setdefault("hyperparameters", {})
    training = config.setdefault("training", {})
    for key in ("epochs", "batch", "imgsz"):
        value = getattr(args, key)
        if value is not None:
            hyperparameters[key] = value
    if args.device is not None:
        training["device"] = args.device
    if args.workers is not None:
        training["workers"] = args.workers
    if args.no_amp:
        training["amp"] = False
    if args.name is not None:
        training["name"] = args.name

    data_config = config.get("data", {})
    dataset_yaml = _project_path(args.data) if args.data else _project_path(data_config["dataset_yaml"])
    if args.data is None and data_config.get("train_split") and data_config.get("val_split"):
        names = {int(class_id): name for class_id, name in data_config["names"].items()}
        create_dataset_yaml(
            train_path=_project_path(data_config["train_split"]),
            val_path=_project_path(data_config["val_split"]),
            test_path=_project_path(data_config["test_split"]) if data_config.get("test_split") else None,
            names=names,
            output_path=dataset_yaml,
            dataset_root=PROJECT_ROOT,
        )

    output_dir = _project_path(args.output or training.get("project", "models/detection"))
    print(f"Stage: {args.stage}")
    print(f"Model: {config.get('model', {}).get('weights', config.get('model', {}).get('name'))}")
    print(f"Dataset: {dataset_yaml}")
    print(f"Output: {output_dir}")
    readiness = validate_training_setup(dataset_yaml, str(training.get("device", 0)))
    print(
        "Ready: "
        f"{readiness['train_images']} train images, "
        f"{readiness['val_images']} validation images, "
        f"{readiness['classes']} classes, device={readiness['device']}"
    )
    if args.check:
        print("Preflight passed; training was not started.")
        return
    train_detector(config, dataset_yaml, output_dir)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
