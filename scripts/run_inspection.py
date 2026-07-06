#!/usr/bin/env python3
"""CLI: Part 1 — Dataset inspection.

Usage:
    python scripts/run_inspection.py
    python scripts/run_inspection.py --config configs/pipeline_config.yaml --datasets configs/datasets.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alpr_dataset.config import PipelineConfig  # noqa: E402
from alpr_dataset.inspection.report import generate_dataset_summary  # noqa: E402
from alpr_dataset.logging_setup import setup_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Part 1 dataset inspection.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "pipeline_config.yaml")
    parser.add_argument("--datasets", type=Path, default=PROJECT_ROOT / "configs" / "datasets.yaml")
    args = parser.parse_args()

    config = PipelineConfig.load(args.config, args.datasets)
    logger = setup_logging(config.logs_dir, name="alpr_dataset")
    logger.info("Starting Part 1: Dataset Inspection")

    dataset_specs = [(d.name, d.root) for d in config.datasets]
    output_dir = config.reports_dir / "eda"

    df = generate_dataset_summary(
        dataset_specs, output_dir, hamming_threshold=config.duplicate_hash_threshold
    )
    logger.info("Inspection complete for %d dataset(s).", len(df))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
