#!/usr/bin/env python3
"""CLI: Part 5 — Configurable preprocessing.

Runs on the harmonized (Part 4) image set by default so preprocessing
operates on a single, consistent naming/annotation scheme regardless of
source dataset.

Usage:
    python scripts/run_preprocessing.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alpr_dataset.config import PipelineConfig  # noqa: E402
from alpr_dataset.io_utils import list_images  # noqa: E402
from alpr_dataset.logging_setup import setup_logging  # noqa: E402
from alpr_dataset.preprocessing.pipeline import PreprocessingPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Part 5 preprocessing.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "pipeline_config.yaml")
    parser.add_argument("--datasets", type=Path, default=PROJECT_ROOT / "configs" / "datasets.yaml")
    parser.add_argument(
        "--preprocessing-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "preprocessing_config.yaml",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Override input image directory (defaults to the harmonized unified/images dir).",
    )
    args = parser.parse_args()

    config = PipelineConfig.load(args.config, args.datasets)
    logger = setup_logging(config.logs_dir, name="alpr_dataset")
    logger.info("Starting Part 5: Preprocessing")

    preprocessing_config = config.preprocessing_config(args.preprocessing_config)
    pipeline = PreprocessingPipeline(preprocessing_config)

    input_dir = args.input_dir or (config.data_processed_dir / "unified" / "images")
    images = list_images(input_dir)
    if not images:
        logger.warning(
            "No images found under %s — run scripts/run_harmonization.py first, "
            "or pass --input-dir explicitly.",
            input_dir,
        )
        return

    output_dir = config.data_processed_dir / "preprocessed"
    comparisons_dir = config.reports_dir / "figures" / "before_after"

    pipeline.run_on_dataset(
        image_paths=images,
        output_dir=output_dir,
        dataset_name="unified",
        comparisons_dir=comparisons_dir,
    )
    logger.info("Part 5 complete -> %s", output_dir)


if __name__ == "__main__":
    main()
