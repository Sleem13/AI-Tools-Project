#!/usr/bin/env python3
"""CLI: Part 6 — Dataset statistics.

Usage:
    python scripts/run_statistics.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alpr_dataset.annotations.loader import load_dataset_annotations  # noqa: E402
from alpr_dataset.config import PipelineConfig  # noqa: E402
from alpr_dataset.inspection.image_stats import batch_compute_stats  # noqa: E402
from alpr_dataset.io_utils import list_images  # noqa: E402
from alpr_dataset.logging_setup import setup_logging  # noqa: E402
from alpr_dataset.statistics.stats_generator import generate_statistics_reports  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Part 6 statistics generation.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "pipeline_config.yaml")
    parser.add_argument("--datasets", type=Path, default=PROJECT_ROOT / "configs" / "datasets.yaml")
    args = parser.parse_args()

    config = PipelineConfig.load(args.config, args.datasets)
    logger = setup_logging(config.logs_dir, name="alpr_dataset")
    logger.info("Starting Part 6: Dataset Statistics")

    per_dataset = {}
    for spec in config.datasets:
        images = list_images(spec.root)
        stats = batch_compute_stats(images)
        annotations = load_dataset_annotations(spec)
        per_dataset[spec.name] = (stats, annotations)

    generate_statistics_reports(per_dataset, config.reports_dir / "statistics")
    logger.info("Part 6 complete.")


if __name__ == "__main__":
    main()
