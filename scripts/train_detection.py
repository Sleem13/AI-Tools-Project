#!/usr/bin/env python3
"""CLI: Train YOLOv8 license plate detection model.

Usage:
    python scripts/train_detection.py
    python scripts/train_detection.py --config configs/model/detection.yaml --epochs 150 --batch 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.detection.trainer import train_yolo


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 license plate detector.")
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "configs" / "model" / "detection.yaml",
                        help="Detection config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size")
    parser.add_argument("--device", type=str, default=None, help="Override device (e.g. '0' or 'cpu')")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models" / "detection")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.epochs is not None:
        config["hyperparameters"]["epochs"] = args.epochs
    if args.batch is not None:
        config["hyperparameters"]["batch"] = args.batch
    if args.imgsz is not None:
        config["hyperparameters"]["imgsz"] = args.imgsz
    if args.device is not None:
        config["training"]["device"] = args.device

    data_cfg = config.get("data", {})

    if data_cfg.get("use_preprocessed", True):
        train_dir = PROJECT_ROOT / data_cfg.get("preprocessed_root", "data/processed/preprocessed") / "images"
    else:
        train_dir = PROJECT_ROOT / data_cfg.get("unified_root", "data/processed/unified") / "images"

    dataset_yaml = PROJECT_ROOT / "configs" / "model" / "dataset.yaml"

    print(f"Training config: {args.config}")
    print(f"Train images: {train_dir}")
    print(f"Output: {args.output}")

    results = train_yolo(
        config=config,
        train_image_dir=train_dir,
        val_image_dir=train_dir,
        output_dir=args.output,
        dataset_yaml_path=dataset_yaml,
    )

    print(f"Training complete. Best model saved in {args.output}")
    if results:
        print(f"Results: {results}")


if __name__ == "__main__":
    main()
