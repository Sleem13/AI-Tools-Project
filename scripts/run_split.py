#!/usr/bin/env python3
"""CLI: Part 7 — Train / validation / test split.

Splits the harmonized dataset (unified/images + unified/labels) by
default so all downstream consumers use one consistent split regardless
of which original dataset each image came from.

Usage:
    python scripts/run_split.py
    python scripts/run_split.py --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alpr_dataset.annotations.loader import load_dataset_annotations  # noqa: E402
from alpr_dataset.config import DatasetSpec, PipelineConfig  # noqa: E402
from alpr_dataset.logging_setup import setup_logging  # noqa: E402
from alpr_dataset.splitting.splitter import (  # noqa: E402
    materialize_split,
    stratified_split,
    write_split_manifests,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Part 7 train/val/test split.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "pipeline_config.yaml")
    parser.add_argument("--datasets", type=Path, default=PROJECT_ROOT / "configs" / "datasets.yaml")
    parser.add_argument(
        "--preprocessing-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "preprocessing_config.yaml",
    )
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--copy-files", action="store_true", help="Physically copy files into split folders.")
    args = parser.parse_args()

    config = PipelineConfig.load(args.config, args.datasets)
    logger = setup_logging(config.logs_dir, name="alpr_dataset")
    logger.info("Starting Part 7: Train/Val/Test Split")

    split_config = config.split_config(args.preprocessing_config)
    if args.train_ratio is not None:
        split_config.train_ratio = args.train_ratio
    if args.val_ratio is not None:
        split_config.val_ratio = args.val_ratio
    if args.test_ratio is not None:
        split_config.test_ratio = args.test_ratio
    if args.copy_files:
        split_config.copy_files = True
    split_config.validate()

    unified_root = config.data_processed_dir / "unified"
    unified_spec = DatasetSpec(
        name="unified",
        root=unified_root,
        annotation_format="yolo",
        image_subdir="images",
        annotation_subdir="labels",
        class_map=config.unified_class_map,
    )
    annotations = load_dataset_annotations(unified_spec)
    if not annotations:
        logger.warning(
            "No harmonized annotations found under %s — run scripts/run_harmonization.py first.",
            unified_root,
        )
        return

    result = stratified_split(annotations, split_config)
    logger.info("Split summary: %s", result.summary())

    manifest_dir = config.reports_dir / "splits"
    write_split_manifests(result, manifest_dir)

    if split_config.copy_files:
        label_lookup = {
            Path(a.image_path): Path(a.annotation_path)
            for a in annotations
            if a.annotation_path is not None
        }
        materialize_split(result, label_lookup, config.data_processed_dir / "split")

    logger.info("Part 7 complete -> %s", manifest_dir)


if __name__ == "__main__":
    main()
