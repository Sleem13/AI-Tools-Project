#!/usr/bin/env python3
"""CLI: Part 2 — Exploratory Data Analysis figure generation.

Usage:
    python scripts/run_eda.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tqdm import tqdm  # noqa: E402

from alpr_dataset.annotations.loader import load_dataset_annotations  # noqa: E402
from alpr_dataset.config import PipelineConfig  # noqa: E402
from alpr_dataset.eda.figures import generate_all_eda_figures  # noqa: E402
from alpr_dataset.inspection.hashing import find_duplicates  # noqa: E402
from alpr_dataset.inspection.image_stats import batch_compute_stats  # noqa: E402
from alpr_dataset.io_utils import list_images  # noqa: E402
from alpr_dataset.logging_setup import setup_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Part 2 EDA figure generation.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "pipeline_config.yaml")
    parser.add_argument("--datasets", type=Path, default=PROJECT_ROOT / "configs" / "datasets.yaml")
    args = parser.parse_args()

    config = PipelineConfig.load(args.config, args.datasets)
    logger = setup_logging(config.logs_dir, name="alpr_dataset")
    logger.info("Starting Part 2: Exploratory Data Analysis")

    dataset_counts = {}
    for spec in config.datasets:
        images = list_images(spec.root)
        dataset_counts[spec.name] = len(images)
        annotations = load_dataset_annotations(spec)
        stats = batch_compute_stats(tqdm(images, desc=f"[{spec.name}] stats"))
        dupes = find_duplicates(images, hamming_threshold=config.duplicate_hash_threshold).near_duplicates

        output_dir = config.reports_dir / "figures" / spec.name
        generate_all_eda_figures(
            dataset_counts={spec.name: len(images)},
            stats=stats,
            annotations=annotations,
            image_paths=images,
            duplicate_pairs=dupes,
            output_dir=output_dir,
            blur_threshold=config.blur_threshold,
        )
        logger.info("EDA figures written for '%s' -> %s", spec.name, output_dir)

    logger.info("Part 2 complete.")


if __name__ == "__main__":
    main()
