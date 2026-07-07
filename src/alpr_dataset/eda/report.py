"""EDA markdown report generator.

Produces `reports/figures/eda_report.md` — a human-readable summary of
every figure produced during Part 2, including key aggregate statistics
and cross-dataset comparisons.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

from alpr_dataset.annotations.schema import ImageAnnotation
from alpr_dataset.inspection.image_stats import ImageStats
from alpr_dataset.io_utils import ensure_dir

logger = logging.getLogger("alpr_dataset")

# Mapping from figure filename (without extension) to human-readable
# description shown in the report.
_FIGURE_DESCRIPTIONS: dict[str, str] = {
    "dataset_size_comparison": "Bar chart comparing the number of images across datasets.",
    "resolution_histograms": "Side-by-side histograms of image width and height.",
    "width_distribution": "Histogram of image widths in pixels.",
    "height_distribution": "Histogram of image heights in pixels.",
    "aspect_ratio_distribution": "Histogram of image aspect ratios (width / height).",
    "brightness_histogram": "Distribution of mean pixel brightness across the dataset.",
    "contrast_histogram": "Distribution of pixel-intensity standard deviation (contrast).",
    "blur_estimation": "Distribution of the variance-of-Laplacian blur metric. Lower values indicate blurrier images.",
    "sharpness_distribution": "Distribution of the mean Sobel gradient magnitude (sharpness).",
    "entropy_distribution": "Distribution of Shannon entropy (bits) across images.",
    "bbox_size_distribution": "Combined histograms of bounding-box width and height.",
    "bbox_width_distribution": "Histogram of bounding-box widths in pixels.",
    "bbox_height_distribution": "Histogram of bounding-box heights in pixels.",
    "bbox_area_distribution": "Histogram of bounding-box areas in square pixels.",
    "bbox_position_heatmap": "2-D heatmap of normalised bounding-box centre positions.",
    "class_distribution": "Bar chart showing the frequency of each class label.",
    "example_images": "Grid of example images from the dataset.",
    "random_samples": "Grid of randomly sampled images.",
    "annotated_samples": "Grid of images with bounding-box annotations overlaid.",
    "color_distribution": "Aggregate RGB colour-channel intensity profiles.",
    "duplicate_visualization": "Side-by-side pairs of near-duplicate images with Hamming distances.",
    "outlier_visualization": "Scatter plot of resolution vs. file size with statistically flagged outliers.",
}


def _stats_summary(stats: list[ImageStats]) -> dict[str, str]:
    valid = [s for s in stats if not s.is_corrupted]
    n = len(valid)
    if n == 0:
        return {"count": "0", "note": "All images corrupted or unreadable."}

    widths = [s.width for s in valid]
    heights = [s.height for s in valid]
    resolutions = [w * h for w, h in zip(widths, heights)]

    brightness = [s.brightness_mean for s in valid]
    contrast = [s.contrast_std for s in valid]
    blur = [s.blur_score for s in valid]
    entropies = [s.entropy for s in valid]

    def _fmt(vals: list[float], decimals: int = 1) -> str:
        if not vals:
            return "N/A"
        return f"{min(vals):.{decimals}f} / {sum(vals)/len(vals):.{decimals}f} / {max(vals):.{decimals}f}"

    return {
        "count": str(n),
        "resolution_range": f"{min(resolutions)} – {max(resolutions)} px",
        "width_range": f"{min(widths)} – {max(widths)} px",
        "height_range": f"{min(heights)} – {max(heights)} px",
        "brightness": _fmt(brightness),
        "contrast": _fmt(contrast),
        "blur": _fmt(blur),
        "entropy": _fmt(entropies, 2),
        "corrupted": str(sum(1 for s in stats if s.is_corrupted)),
    }


def _bbox_summary(annotations: list[ImageAnnotation]) -> dict[str, str]:
    n_images_with_boxes = sum(1 for a in annotations if a.boxes)
    n_boxes = sum(len(a.boxes) for a in annotations)
    return {
        "images_with_boxes": str(n_images_with_boxes),
        "total_boxes": str(n_boxes),
    }


def _duplicate_summary(duplicate_pairs: list[tuple[Path, Path, int]]) -> dict[str, str]:
    return {
        "near_duplicate_pairs": str(len(duplicate_pairs)),
    }


def generate_eda_report(
    dataset_name: str,
    stats: list[ImageStats],
    annotations: list[ImageAnnotation],
    image_paths: list[Path],
    duplicate_pairs: list[tuple[Path, Path, int]],
    figures_dir: Path,
    output_dir: Path,
) -> None:
    """Generate a markdown EDA report and write it to ``output_dir``."""
    ensure_dir(output_dir)

    stat_summary = _stats_summary(stats)
    bbox_summary = _bbox_summary(annotations)
    dup_summary = _duplicate_summary(duplicate_pairs)

    lines: list[str] = []
    lines.append(f"# Exploratory Data Analysis — {dataset_name}")
    lines.append("")
    lines.append(f"**Generated:** automatically by `scripts/run_eda.py`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Dataset Overview")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total images scanned | {stat_summary['count']} |")
    lines.append(f"| Corrupted / unreadable | {stat_summary['corrupted']} |")
    lines.append(f"| Width range | {stat_summary['width_range']} |")
    lines.append(f"| Height range | {stat_summary['height_range']} |")
    lines.append(f"| Resolution range | {stat_summary['resolution_range']} |")
    lines.append(f"| Brightness (min / mean / max) | {stat_summary['brightness']} |")
    lines.append(f"| Contrast (min / mean / max) | {stat_summary['contrast']} |")
    lines.append(f"| Blur score (min / mean / max) | {stat_summary['blur']} |")
    lines.append(f"| Entropy (min / mean / max) | {stat_summary['entropy']} bits |")
    lines.append("")
    lines.append("### Annotations")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Images with bounding boxes | {bbox_summary['images_with_boxes']} |")
    lines.append(f"| Total bounding boxes | {bbox_summary['total_boxes']} |")
    lines.append(f"| Near-duplicate pairs found | {dup_summary['near_duplicate_pairs']} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Generated Figures")
    lines.append("")
    lines.append("The following figures have been saved as PNG (raster) and SVG (vector) pairs")
    lines.append(f"in `{figures_dir.relative_to(output_dir.parent.parent) if figures_dir.is_relative_to(output_dir.parent.parent) else figures_dir}`.")
    lines.append("")

    # List each figure with description and file links
    figure_names: list[str] = []
    if figures_dir.exists():
        for f in sorted(figures_dir.glob("*.png")):
            stem = f.stem
            if stem not in figure_names:
                figure_names.append(stem)

    for name in sorted(figure_names):
        desc = _FIGURE_DESCRIPTIONS.get(name, "EDA figure.")
        lines.append(f"### {name.replace('_', ' ').title()}")
        lines.append("")
        lines.append(f"{desc}")
        lines.append("")
        lines.append(f"- PNG: `{name}.png`")
        lines.append(f"- SVG: `{name}.svg`")
        lines.append("")

    # Per-figure statistics annotations
    lines.append("---")
    lines.append("")
    lines.append("## Key Observations")
    lines.append("")

    valid = [s for s in stats if not s.is_corrupted]
    if valid:
        widths = [s.width for s in valid]
        heights = [s.height for s in valid]
        ratios = [w / h for w, h in zip(widths, heights) if h > 0]
        brightness = [s.brightness_mean for s in valid]
        blur = [s.blur_score for s in valid]

        if widths:
            lines.append(f"- **Widths** range from {min(widths)} px to {max(widths)} px.")
        if heights:
            lines.append(f"- **Heights** range from {min(heights)} px to {max(heights)} px.")
        if ratios:
            lines.append(f"- **Aspect ratios** range from {min(ratios):.3f} to {max(ratios):.3f}.")
        if brightness:
            low_brightness = sum(1 for b in brightness if b < 50)
            high_brightness = sum(1 for b in brightness if b > 200)
            lines.append(f"- **Brightness**: {low_brightness} images are very dark (< 50), {high_brightness} are very bright (> 200).")
        if blur:
            blurry = sum(1 for b in blur if b < 100)
            lines.append(f"- **Blur**: {blurry} images fall below the blur threshold (100).")

    boxes = [b for a in annotations for b in a.boxes]
    if boxes:
        areas = [b.area for b in boxes if b.area > 0]
        if areas:
            lines.append(f"- **Bounding box areas** range from {min(areas):.0f} px² to {max(areas):.0f} px².")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Report automatically generated by the ALPR dataset EDA pipeline.*")
    lines.append("")

    report_path = output_dir / f"{dataset_name}_eda_report.md"
    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("EDA report written: %s", report_path)
