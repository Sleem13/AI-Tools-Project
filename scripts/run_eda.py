#!/usr/bin/env python3
"""CLI: Part 2 — Exploratory Data Analysis figure generation.

Produces per-dataset and cross-dataset figures, plus an EDA markdown
report summarising every output.  All figures are saved as PNG + SVG
pairs under ``reports/figures/``.

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

from tqdm import tqdm

from alpr_dataset.annotations.loader import load_dataset_annotations
from alpr_dataset.config import PipelineConfig
from alpr_dataset.eda.figures import generate_all_eda_figures
from alpr_dataset.eda.report import generate_eda_report
from alpr_dataset.inspection.hashing import find_duplicates
from alpr_dataset.inspection.image_stats import batch_compute_stats
from alpr_dataset.io_utils import list_images
from alpr_dataset.logging_setup import setup_logging


def _validate_outputs(figures_dir: Path, expected_stems: set[str]) -> list[str]:
    """Check that every expected figure exists as both PNG and SVG."""
    missing: list[str] = []
    if not figures_dir.exists():
        return list(expected_stems)
    for stem in expected_stems:
        png_ok = (figures_dir / f"{stem}.png").is_file()
        svg_ok = (figures_dir / f"{stem}.svg").is_file()
        if not png_ok or not svg_ok:
            missing.append(stem)
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Part 2 EDA figure generation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pipeline_config.yaml",
    )
    parser.add_argument(
        "--datasets",
        type=Path,
        default=PROJECT_ROOT / "configs" / "datasets.yaml",
    )
    args = parser.parse_args()

    config = PipelineConfig.load(args.config, args.datasets)
    logger = setup_logging(config.logs_dir, name="alpr_dataset")
    logger.info("Starting Part 2: Exploratory Data Analysis")

    # ------------------------------------------------------------------ #
    # 1. Collect data from every dataset
    # ------------------------------------------------------------------ #
    all_data: dict[str, dict] = {}
    all_stats: list = []
    all_annotations: list = []
    all_image_paths: list[Path] = []
    all_dupes: list[tuple[Path, Path, int]] = []

    for spec in config.datasets:
        images = list_images(spec.root)
        logger.info("[%s] Scanning %d images …", spec.name, len(images))

        annotations = load_dataset_annotations(spec)
        stats = batch_compute_stats(tqdm(images, desc=f"[{spec.name}] stats"))
        dupes = find_duplicates(images, hamming_threshold=config.duplicate_hash_threshold).near_duplicates

        all_data[spec.name] = {
            "images": images,
            "stats": stats,
            "annotations": annotations,
            "dupes": dupes,
        }
        all_stats.extend(stats)
        all_annotations.extend(annotations)
        all_image_paths.extend(images)
        all_dupes.extend(dupes)

    # ------------------------------------------------------------------ #
    # 2. Cross-dataset comparison figure in the shared figures directory
    # ------------------------------------------------------------------ #
    shared_fig_dir = config.reports_dir / "figures"
    dataset_counts = {name: len(d["images"]) for name, d in all_data.items()}
    logger.info("Generating cross-dataset comparison …")
    # Import here to avoid circular dependency at module level
    from alpr_dataset.eda.figures import plot_dataset_size_comparison

    plot_dataset_size_comparison(dataset_counts, shared_fig_dir)

    # ------------------------------------------------------------------ #
    # 3. Per-dataset figures + EDA reports
    # ------------------------------------------------------------------ #
    expected_stems: set[str] = {
        "resolution_histograms",
        "width_distribution",
        "height_distribution",
        "aspect_ratio_distribution",
        "brightness_histogram",
        "contrast_histogram",
        "blur_estimation",
        "sharpness_distribution",
        "entropy_distribution",
        "bbox_size_distribution",
        "bbox_width_distribution",
        "bbox_height_distribution",
        "bbox_area_distribution",
        "bbox_position_heatmap",
        "class_distribution",
        "example_images",
        "random_samples",
        "annotated_samples",
        "color_distribution",
        "duplicate_visualization",
        "outlier_visualization",
    }

    all_missing: list[str] = []

    for spec in config.datasets:
        d = all_data[spec.name]
        output_dir = shared_fig_dir / spec.name

        generate_all_eda_figures(
            dataset_counts={spec.name: len(d["images"])},
            stats=d["stats"],
            annotations=d["annotations"],
            image_paths=d["images"],
            duplicate_pairs=d["dupes"],
            output_dir=output_dir,
            blur_threshold=config.blur_threshold,
        )

        # Per-dataset EDA report
        generate_eda_report(
            dataset_name=spec.name,
            stats=d["stats"],
            annotations=d["annotations"],
            image_paths=d["images"],
            duplicate_pairs=d["dupes"],
            figures_dir=output_dir,
            output_dir=shared_fig_dir,
        )

        # Validate per-dataset figures
        missing = _validate_outputs(output_dir, expected_stems)
        if missing:
            logger.warning("[%s] Missing figures: %s", spec.name, missing)
            all_missing.extend(f"{spec.name}/{m}" for m in missing)
        else:
            logger.info("[%s] All %d figures validated.", spec.name, len(expected_stems))

    # Validate cross-dataset figures
    cross_missing = _validate_outputs(shared_fig_dir, {"dataset_size_comparison"})
    if cross_missing:
        all_missing.extend(cross_missing)

    # ------------------------------------------------------------------ #
    # 4. Summary
    # ------------------------------------------------------------------ #
    if all_missing:
        logger.error(
            "VALIDATION FAILED — %d figure(s) missing: %s",
            len(all_missing),
            all_missing,
        )
        sys.exit(1)
    else:
        total_expected = len(config.datasets) * (len(expected_stems) - 1) + 1
        logger.info(
            "Part 2 complete — all %d figures validated (PNG + SVG).",
            total_expected,
        )


if __name__ == "__main__":
    main()
