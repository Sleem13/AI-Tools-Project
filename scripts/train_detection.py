#!/usr/bin/env python3
"""Train a YOLO11 vehicle or license-plate detection stage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.data import create_dataset_yaml  # noqa: E402
from src.detection.trainer import train_detector  # noqa: E402

DEFAULT_CONFIGS = {
    "plate": PROJECT_ROOT / "configs" / "model" / "detection.yaml",
    "vehicle": PROJECT_ROOT / "configs" / "model" / "vehicle_detection.yaml",
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO11 stage for the vehicle-to-plate cascade.")
    parser.add_argument("--stage", choices=("vehicle", "plate"), default="plate")
    parser.add_argument("--config", type=Path, default=None, help="Override the stage configuration")
    parser.add_argument("--data", type=Path, default=None, help="Use an existing Ultralytics dataset YAML")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="GPU id such as 0, or cpu")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--name", type=str, default=None, help="Override the Ultralytics run name")
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
    if args.name is not None:
        training["name"] = args.name

    data_config = config.get("data", {})
    dataset_yaml = _project_path(args.data) if args.data else _project_path(data_config["dataset_yaml"])
    if args.data is None:
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
    train_detector(config, dataset_yaml, output_dir)


if __name__ == "__main__":
    main()
