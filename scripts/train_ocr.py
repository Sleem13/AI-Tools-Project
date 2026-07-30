#!/usr/bin/env python3
"""CLI: Train CRNN license plate OCR model.

Usage:
    python scripts/train_ocr.py
    python scripts/train_ocr.py --config configs/model/ocr.yaml --epochs 100 --batch 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from src.ocr.data import OCRPlateDataset, build_vocab
from src.ocr.trainer import train_crnn


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CRNN license plate OCR.")
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "configs" / "model" / "ocr.yaml",
                        help="OCR config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models" / "ocr")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.epochs is not None:
        config["hyperparameters"]["epochs"] = args.epochs
    if args.batch is not None:
        config["hyperparameters"]["batch"] = args.batch
    if args.device is not None:
        config["training"]["device"] = args.device

    data_cfg = config.get("data", {})
    vocab = build_vocab()

    annotation_path = PROJECT_ROOT / data_cfg.get("annotations_file", "data/ocr/labels.json")
    crops_root = PROJECT_ROOT / data_cfg.get("crops_root", "data/ocr/crops")

    dataset = OCRPlateDataset(
        annotation_path=annotation_path,
        image_root=crops_root,
        vocab=vocab,
        input_height=config.get("model", {}).get("input_height", 32),
        input_width=config.get("model", {}).get("input_width", 128),
    )

    print(f"Training config: {args.config}")
    print(f"OCR samples: {len(dataset)}")
    print(f"Output: {args.output}")

    results = train_crnn(
        config=config,
        train_dataset=dataset,
        val_dataset=None,
        output_dir=args.output,
    )

    print(f"Training complete. Best model -> {results.get('best_path')}")


if __name__ == "__main__":
    main()
