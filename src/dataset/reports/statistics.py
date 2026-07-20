from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..inspector.metadata import ImageMetadata
from ..inspector.scanner import ScanResult


@dataclass
class DatasetStats:
    dataset_name: str
    image_count: int = 0
    annotation_count: int = 0
    widths: list[int] = field(default_factory=list)
    heights: list[int] = field(default_factory=list)
    resolutions: list[int] = field(default_factory=list)
    file_sizes: list[int] = field(default_factory=list)
    aspect_ratios: list[float] = field(default_factory=list)
    brightnesses: list[float] = field(default_factory=list)
    contrasts: list[float] = field(default_factory=list)
    sharpnesses: list[float] = field(default_factory=list)
    blur_scores: list[float] = field(default_factory=list)

    @property
    def avg_resolution(self) -> float | None:
        return _safe_mean(self.resolutions)

    @property
    def median_resolution(self) -> float | None:
        return _safe_median(self.resolutions)

    @property
    def min_resolution(self) -> int | None:
        return _safe_min(self.resolutions)

    @property
    def max_resolution(self) -> int | None:
        return _safe_max(self.resolutions)

    @property
    def avg_width(self) -> float | None:
        return _safe_mean(self.widths)

    @property
    def avg_height(self) -> float | None:
        return _safe_mean(self.heights)

    @property
    def avg_file_size(self) -> float | None:
        return _safe_mean(self.file_sizes)

    @property
    def median_file_size(self) -> float | None:
        return _safe_median(self.file_sizes)

    @property
    def min_file_size(self) -> int | None:
        return _safe_min(self.file_sizes)

    @property
    def max_file_size(self) -> int | None:
        return _safe_max(self.file_sizes)

    @property
    def avg_brightness(self) -> float | None:
        return _safe_mean(self.brightnesses)

    @property
    def min_brightness(self) -> float | None:
        return _safe_min(self.brightnesses)

    @property
    def max_brightness(self) -> float | None:
        return _safe_max(self.brightnesses)

    @property
    def avg_contrast(self) -> float | None:
        return _safe_mean(self.contrasts)

    @property
    def avg_sharpness(self) -> float | None:
        return _safe_mean(self.sharpnesses)

    @property
    def avg_blur_score(self) -> float | None:
        return _safe_mean(self.blur_scores)

    @property
    def avg_aspect_ratio(self) -> float | None:
        return _safe_mean(self.aspect_ratios)


def _safe_mean(values: list) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _safe_median(values: list) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _safe_min(values: list) -> int | None:
    if not values:
        return None
    return int(np.min(values))


def _safe_max(values: list) -> int | None:
    if not values:
        return None
    return int(np.max(values))


def compute_stats(
    dataset_name: str,
    metadata: list[ImageMetadata],
    scan_result: ScanResult,
) -> DatasetStats:
    stats = DatasetStats(
        dataset_name=dataset_name,
        image_count=scan_result.total_images,
        annotation_count=scan_result.total_annotations,
    )

    for m in metadata:
        if m.width is not None and m.height is not None:
            stats.widths.append(m.width)
            stats.heights.append(m.height)
            stats.resolutions.append(m.width * m.height)
            stats.aspect_ratios.append(m.width / m.height if m.height > 0 else 0.0)

        stats.file_sizes.append(m.size_bytes)

        if m.brightness is not None:
            stats.brightnesses.append(m.brightness)
        if m.contrast is not None:
            stats.contrasts.append(m.contrast)
        if m.sharpness is not None:
            stats.sharpnesses.append(m.sharpness)
        if m.blur_score is not None:
            stats.blur_scores.append(m.blur_score)

    return stats


class StatisticsReportGenerator:
    @staticmethod
    def generate_csv(
        stats_list: list[DatasetStats],
        output_path: Path,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "dataset",
                "image_count",
                "annotation_count",
                "avg_width",
                "avg_height",
                "avg_resolution",
                "median_resolution",
                "min_resolution",
                "max_resolution",
                "avg_file_size",
                "median_file_size",
                "min_file_size",
                "max_file_size",
                "avg_brightness",
                "min_brightness",
                "max_brightness",
                "avg_contrast",
                "avg_sharpness",
                "avg_blur_score",
                "avg_aspect_ratio",
            ])

            for s in stats_list:
                writer.writerow([
                    s.dataset_name,
                    s.image_count,
                    s.annotation_count,
                    _fmt(s.avg_width),
                    _fmt(s.avg_height),
                    _fmt(s.avg_resolution),
                    _fmt(s.median_resolution),
                    _fmt(s.min_resolution),
                    _fmt(s.max_resolution),
                    _fmt(s.avg_file_size),
                    _fmt(s.median_file_size),
                    _fmt(s.min_file_size),
                    _fmt(s.max_file_size),
                    _fmt(s.avg_brightness),
                    _fmt(s.min_brightness),
                    _fmt(s.max_brightness),
                    _fmt(s.avg_contrast),
                    _fmt(s.avg_sharpness),
                    _fmt(s.avg_blur_score),
                    _fmt(s.avg_aspect_ratio),
                ])

        return output_path

    @staticmethod
    def generate_markdown(
        stats_list: list[DatasetStats],
        output_path: Path,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_images = sum(s.image_count for s in stats_list)
        total_annotations = sum(s.annotation_count for s in stats_list)

        lines: list[str] = [
            "# Dataset Statistics Report",
            "",
            "## Summary",
            "",
            f"- **Datasets:** {len(stats_list)}",
            f"- **Total Images:** {total_images}",
            f"- **Total Annotations:** {total_annotations}",
            "",
            "---",
            "",
        ]

        for s in stats_list:
            lines.append(f"## {s.dataset_name}")
            lines.append("")
            lines.append("### Counts")
            lines.append("")
            lines.append(f"- **Images:** {s.image_count}")
            lines.append(f"- **Annotations:** {s.annotation_count}")
            lines.append("")

            lines.append("### Resolution")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Average Width | {_fmt(s.avg_width)} px |")
            lines.append(f"| Average Height | {_fmt(s.avg_height)} px |")
            lines.append(f"| Average Resolution | {_fmt(s.avg_resolution)} px² |")
            lines.append(f"| Median Resolution | {_fmt(s.median_resolution)} px² |")
            lines.append(f"| Min Resolution | {_fmt(s.min_resolution)} px² |")
            lines.append(f"| Max Resolution | {_fmt(s.max_resolution)} px² |")
            lines.append(f"| Avg Aspect Ratio | {_fmt(s.avg_aspect_ratio)} |")
            lines.append("")

            lines.append("### File Size")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Average | {_fmt_bytes(s.avg_file_size)} |")
            lines.append(f"| Median | {_fmt_bytes(s.median_file_size)} |")
            lines.append(f"| Min | {_fmt_bytes(s.min_file_size)} |")
            lines.append(f"| Max | {_fmt_bytes(s.max_file_size)} |")
            lines.append("")

            lines.append("### Brightness")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Average | {_fmt(s.avg_brightness)} |")
            lines.append(f"| Min | {_fmt(s.min_brightness)} |")
            lines.append(f"| Max | {_fmt(s.max_brightness)} |")
            lines.append("")

            lines.append("### Image Quality")
            lines.append("")
            lines.append("| Metric | Average Value |")
            lines.append("|--------|---------------|")
            lines.append(f"| Contrast | {_fmt(s.avg_contrast)} |")
            lines.append(f"| Sharpness | {_fmt(s.avg_sharpness)} |")
            lines.append(f"| Blur Score | {_fmt(s.avg_blur_score)} |")
            lines.append("")
            lines.append("---")
            lines.append("")

        lines.append("*Report generated automatically by Dataset Inspector.*")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path


def _fmt(val: object) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _fmt_bytes(val: object) -> str:
    if val is None:
        return ""
    if not isinstance(val, (int, float)):
        return str(val)
    b = float(val)
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.2f} {unit}"
        b /= 1024
    return f"{b:.2f} TB"
