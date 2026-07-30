"""Part 6 — Dataset statistics.

Produces number of images/labels, average boxes/image, largest/smallest
image, median dimensions, standard deviations, a resolution table, and
annotation statistics. Writes CSV + JSON + Markdown into reports/statistics/.
"""

from __future__ import annotations

import json
import logging
import statistics as pystats
from pathlib import Path

import pandas as pd

from alpr_dataset.annotations.schema import ImageAnnotation
from alpr_dataset.inspection.image_stats import ImageStats
from alpr_dataset.io_utils import ensure_dir

logger = logging.getLogger("alpr_dataset")


def compute_dataset_statistics(
    dataset_name: str,
    stats: list[ImageStats],
    annotations: list[ImageAnnotation],
) -> dict:
    valid = [s for s in stats if not s.is_corrupted]
    widths = [s.width for s in valid]
    heights = [s.height for s in valid]
    resolutions = [w * h for w, h in zip(widths, heights, strict=False)]
    file_sizes = [s.file_size_bytes for s in valid]
    boxes_per_image = [a.n_boxes for a in annotations]

    def _safe_stat(fn, data):
        return fn(data) if data else 0

    largest = max(valid, key=lambda s: s.width * s.height, default=None)
    smallest = min(valid, key=lambda s: s.width * s.height, default=None)

    stats_dict = {
        "dataset_name": dataset_name,
        "n_images": len(valid),
        "n_labels": sum(1 for a in annotations if a.n_boxes > 0),
        "n_total_boxes": sum(boxes_per_image),
        "avg_boxes_per_image": round(_safe_stat(pystats.mean, boxes_per_image), 4),
        "median_boxes_per_image": _safe_stat(pystats.median, boxes_per_image),
        "largest_image": {
            "path": str(largest.path),
            "width": largest.width,
            "height": largest.height,
        }
        if largest
        else None,
        "smallest_image": {
            "path": str(smallest.path),
            "width": smallest.width,
            "height": smallest.height,
        }
        if smallest
        else None,
        "median_width": _safe_stat(pystats.median, widths),
        "median_height": _safe_stat(pystats.median, heights),
        "std_width": round(_safe_stat(pystats.pstdev, widths), 3) if len(widths) > 1 else 0,
        "std_height": round(_safe_stat(pystats.pstdev, heights), 3) if len(heights) > 1 else 0,
        "std_resolution": round(_safe_stat(pystats.pstdev, resolutions), 3)
        if len(resolutions) > 1
        else 0,
        "avg_file_size_kb": round(_safe_stat(pystats.mean, file_sizes) / 1024, 2)
        if file_sizes
        else 0,
        "std_file_size_kb": round(_safe_stat(pystats.pstdev, file_sizes) / 1024, 2)
        if len(file_sizes) > 1
        else 0,
    }
    return stats_dict


def build_resolution_table(stats: list[ImageStats]) -> pd.DataFrame:
    valid = [s for s in stats if not s.is_corrupted]
    df = pd.DataFrame(
        [
            {
                "path": str(s.path),
                "width": s.width,
                "height": s.height,
                "aspect_ratio": round(s.width / s.height, 4) if s.height else 0,
                "channels": s.channels,
                "file_size_kb": round(s.file_size_bytes / 1024, 2),
            }
            for s in valid
        ]
    )
    return df


def generate_statistics_reports(
    per_dataset: dict[str, tuple[list[ImageStats], list[ImageAnnotation]]],
    output_dir: Path,
) -> None:
    """Generate Part 6 outputs for every dataset plus a combined summary.

    Args:
        per_dataset: {dataset_name: (image_stats, annotations)}
        output_dir: reports/statistics/
    """
    output_dir = ensure_dir(output_dir)
    all_summaries = []

    for name, (stats, annotation_records) in per_dataset.items():
        summary = compute_dataset_statistics(name, stats, annotation_records)
        all_summaries.append(summary)

        res_table = build_resolution_table(stats)
        res_table.to_csv(output_dir / f"{name}_resolution_table.csv", index=False)

        (output_dir / f"{name}_statistics.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    combined_df = pd.DataFrame(
        [
            {k: v for k, v in s.items() if not isinstance(v, dict)}
            for s in all_summaries
        ]
    )
    combined_df.to_csv(output_dir / "combined_statistics.csv", index=False)

    md_lines = ["# Dataset Statistics Report", ""]
    for s in all_summaries:
        md_lines.append(f"## {s['dataset_name']}")
        md_lines.append(f"- Images: {s['n_images']}")
        md_lines.append(f"- Labeled images: {s['n_labels']}")
        md_lines.append(f"- Total boxes: {s['n_total_boxes']}")
        md_lines.append(f"- Avg boxes/image: {s['avg_boxes_per_image']}")
        md_lines.append(f"- Median boxes/image: {s['median_boxes_per_image']}")
        if s["largest_image"]:
            li = s["largest_image"]
            md_lines.append(f"- Largest image: {li['width']}x{li['height']} ({li['path']})")
        if s["smallest_image"]:
            si = s["smallest_image"]
            md_lines.append(f"- Smallest image: {si['width']}x{si['height']} ({si['path']})")
        md_lines.append(f"- Median dimensions: {s['median_width']}x{s['median_height']}")
        md_lines.append(
            f"- Std dev (width/height/resolution): "
            f"{s['std_width']}/{s['std_height']}/{s['std_resolution']}"
        )
        md_lines.append(
            f"- File size (KB): avg {s['avg_file_size_kb']}, std {s['std_file_size_kb']}"
        )
        md_lines.append("")

    (output_dir / "statistics_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Wrote statistics reports -> %s", output_dir)
