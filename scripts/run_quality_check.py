#!/usr/bin/env python3
"""CLI: Part 3 — Data quality assessment.

Usage:
    python scripts/run_quality_check.py
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
from alpr_dataset.eda.quality import build_quality_report  # noqa: E402
from alpr_dataset.io_utils import list_images  # noqa: E402
from alpr_dataset.logging_setup import setup_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Part 3 data quality assessment.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "pipeline_config.yaml")
    parser.add_argument("--datasets", type=Path, default=PROJECT_ROOT / "configs" / "datasets.yaml")
    args = parser.parse_args()

    config = PipelineConfig.load(args.config, args.datasets)
    logger = setup_logging(config.logs_dir, name="alpr_dataset")
    logger.info("Starting Part 3: Data Quality Assessment")

    output_dir = config.reports_dir / "quality"
    valid_class_ids = set(config.unified_class_map.keys()) or None

    for spec in config.datasets:
        images = list_images(spec.root)
        annotations = load_dataset_annotations(spec)
        build_quality_report(
            dataset_name=spec.name,
            annotations=annotations,
            image_paths=images,
            output_dir=output_dir,
            valid_class_ids=valid_class_ids,
            hamming_threshold=config.duplicate_hash_threshold,
        )

    logger.info("Part 3 complete -> %s", output_dir)


if __name__ == "__main__":
    main()
