#!/usr/bin/env python3
"""CLI: Part 4 — Dataset harmonization.

Usage:
    python scripts/run_harmonization.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alpr_dataset.annotations.loader import load_dataset_annotations
from alpr_dataset.config import PipelineConfig
from alpr_dataset.harmonization.harmonizer import (
    harmonize_dataset,
    write_harmonization_metadata,
)
from alpr_dataset.logging_setup import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Part 4 dataset harmonization.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "pipeline_config.yaml")
    parser.add_argument("--datasets", type=Path, default=PROJECT_ROOT / "configs" / "datasets.yaml")
    args = parser.parse_args()

    config = PipelineConfig.load(args.config, args.datasets)
    logger = setup_logging(config.logs_dir, name="alpr_dataset")
    logger.info("Starting Part 4: Dataset Harmonization")

    unified_root = config.data_processed_dir / "unified"
    all_records = []

    for spec in config.datasets:
        annotations = load_dataset_annotations(spec)
        records = harmonize_dataset(
            dataset_name=spec.name,
            annotations=annotations,
            unified_class_map=config.unified_class_map,
            local_class_map=spec.class_map,
            output_root=unified_root,
        )
        all_records.extend(records)
        logger.info("Harmonized %d files for '%s'", len(records), spec.name)

    write_harmonization_metadata(all_records, config.reports_dir / "harmonization")
    logger.info("Part 4 complete -> %s", unified_root)


if __name__ == "__main__":
    main()
