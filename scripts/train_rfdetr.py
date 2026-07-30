#!/usr/bin/env python3
"""Convert YOLO labels to RF-DETR COCO format and optionally fine-tune RF-DETR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.rfdetr import convert_yolo_dataset_to_rfdetr_coco, train_rfdetr

DEFAULT_CONFIGS = {
    "plate": PROJECT_ROOT / "configs" / "model" / "rfdetr_plate.yaml",
    "character": PROJECT_ROOT / "configs" / "model" / "rfdetr_character.yaml",
}


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune RF-DETR on this project's labeled boxes.")
    parser.add_argument("--stage", choices=tuple(DEFAULT_CONFIGS), default="plate")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=None, help="Override source YOLO data.yaml")
    parser.add_argument("--coco-dir", type=Path, default=None, help="Override converted RF-DETR dataset directory")
    parser.add_argument("--output", type=Path, default=None, help="Override RF-DETR training output directory")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=None, help="RF-DETR resolution; must be divisible by 56")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--convert-only", action="store_true", help="Only create the COCO dataset; do not train")
    parser.add_argument("--no-copy-images", action="store_true", help="Reference absolute image paths instead of copying")
    args = parser.parse_args()

    config = load_config(_project_path(args.config or DEFAULT_CONFIGS[args.stage]))
    data_config = config.setdefault("data", {})
    training = config.setdefault("training", {})
    if args.output is not None:
        training["output_dir"] = str(_project_path(args.output))
    for key, arg_name in {
        "epochs": "epochs",
        "batch_size": "batch_size",
        "grad_accum_steps": "grad_accum_steps",
        "resolution": "resolution",
        "device": "device",
    }.items():
        value = getattr(args, arg_name)
        if value is not None:
            training[key] = value

    data_yaml = _project_path(args.data or data_config["source_yaml"])
    coco_dir = _project_path(args.coco_dir or data_config["coco_dir"])
    summary = convert_yolo_dataset_to_rfdetr_coco(
        data_yaml,
        coco_dir,
        copy_images=not args.no_copy_images,
    )
    print(f"Stage: {args.stage}")
    print(f"RF-DETR dataset: {summary.output_dir}")
    for split in ("train", "valid", "test"):
        if split in summary.images:
            print(f"  {split}: {summary.images[split]} images, {summary.annotations.get(split, 0)} boxes")
    print(f"  categories: {summary.categories}")
    if args.convert_only:
        print("Conversion complete; training was not started.")
        return
    train_rfdetr(config, summary.output_dir)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
