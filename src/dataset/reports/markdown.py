from __future__ import annotations

from pathlib import Path

from ..inspector.metadata import ImageMetadata
from ..inspector.scanner import ScanResult
from ..inspector.validator import ValidationSummary


class MarkdownReportGenerator:
    @staticmethod
    def generate_inventory_report(
        scan_results: list[ScanResult],
        metadata_list: list[list[ImageMetadata]],
        tree_texts: list[str],
        output_path: Path,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = [
            "# Dataset Inventory Report",
            "",
            f"Generated for **{len(scan_results)}** dataset(s).",
            "",
            "---",
            "",
        ]

        for i, (scan, metadatas, tree) in enumerate(zip(scan_results, metadata_list, tree_texts)):
            lines.append(f"## {i + 1}. {scan.dataset_name}")
            lines.append("")
            lines.append("### Overview")
            lines.append("")
            lines.append(f"- **Path:** `{scan.root_path}`")
            lines.append(f"- **Total Images:** {scan.total_images}")
            lines.append(f"- **Total Annotations:** {scan.total_annotations}")
            lines.append(f"- **Annotation Format:** `{scan.annotation_format.value}`")
            lines.append(f"- **Orphan Images:** {len(scan.orphan_images)}")
            lines.append(f"- **Orphan Annotations:** {len(scan.orphan_annotations)}")
            lines.append("")

            if scan.image_formats:
                lines.append("### Image Formats")
                lines.append("")
                lines.append("| Format | Count |")
                lines.append("|--------|-------|")
                for fmt, count in sorted(scan.image_formats.items()):
                    lines.append(f"| {fmt} | {count} |")
                lines.append("")

            if scan.annotation_formats:
                lines.append("### Annotation Formats")
                lines.append("")
                lines.append("| Format | Count |")
                lines.append("|--------|-------|")
                for fmt, count in sorted(scan.annotation_formats.items()):
                    lines.append(f"| {fmt} | {count} |")
                lines.append("")

            lines.append("### Directory Tree")
            lines.append("")
            lines.append("```")
            lines.append(tree)
            lines.append("```")
            lines.append("")

            lines.append("### Image Metadata (first 10 samples)")
            lines.append("")
            lines.append("| # | Filename | Width | Height | Mode | Size (KB) | MD5 |")
            lines.append("|---|----------|-------|--------|------|-----------|-----|")
            for j, meta in enumerate(metadatas[:10]):
                w = str(meta.width) if meta.width is not None else "-"
                h = str(meta.height) if meta.height is not None else "-"
                mode = meta.mode if meta.mode is not None else "-"
                size_kb = f"{meta.size_bytes / 1024:.1f}"
                md5_short = (meta.md5 or "-")[:16]
                lines.append(f"| {j + 1} | {meta.filename} | {w} | {h} | {mode} | {size_kb} | {md5_short} |")
            lines.append("")
            lines.append("---")
            lines.append("")

        lines.append("*Report generated automatically by Dataset Inspector.*")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    @staticmethod
    def generate_quality_report(
        validation_summaries: list[tuple[str, ValidationSummary]],
        tree_texts: list[str],
        output_path: Path,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_images = sum(s.total_images for _, s in validation_summaries)
        total_annotations = sum(s.total_annotations for _, s in validation_summaries)
        total_issues = sum(s.total_issues for _, s in validation_summaries)

        lines: list[str] = [
            "# Dataset Quality Report",
            "",
            "## Summary",
            "",
            f"- **Datasets:** {len(validation_summaries)}",
            f"- **Total Images:** {total_images}",
            f"- **Total Annotations:** {total_annotations}",
            f"- **Total Issues Found:** {total_issues}",
            "",
            "---",
            "",
        ]

        for i, (dataset_name, summary) in enumerate(validation_summaries):
            lines.append(f"## {i + 1}. {dataset_name}")
            lines.append("")
            lines.append(f"- **Images:** {summary.total_images}")
            lines.append(f"- **Annotations:** {summary.total_annotations}")
            lines.append(f"- **Total Issues:** {summary.total_issues}")
            lines.append("")

            categories = [
                ("Corrupted Images", summary.corrupted_images),
                ("Unreadable Images", summary.unreadable_images),
                ("Zero-Byte Files", summary.zero_byte_files),
                ("Unsupported Extensions", summary.unsupported_extensions),
                ("Missing Labels", summary.missing_labels),
                ("Orphan Labels", summary.orphan_labels),
                ("Empty Annotations", summary.empty_annotations),
                ("Invalid Bounding Boxes", summary.invalid_bboxes),
                ("Duplicate Filenames", summary.duplicate_filenames),
                ("Duplicate Images (MD5)", summary.duplicate_images),
                ("Near-Duplicate Images", summary.near_duplicates),
            ]

            lines.append("| Issue Category | Count |")
            lines.append("|----------------|-------|")
            for cat_name, cat_issues in categories:
                lines.append(f"| {cat_name} | {len(cat_issues)} |")
            lines.append("")

            for cat_name, cat_issues in categories:
                if cat_issues:
                    lines.append(f"### {cat_name} ({len(cat_issues)})")
                    lines.append("")
                    for issue in cat_issues:
                        lines.append(f"- `{issue.file_path}`: {issue.message}")
                    lines.append("")

            if i < len(validation_summaries) - 1:
                lines.append("---")
                lines.append("")

        lines.append("*Report generated automatically by Dataset Inspector.*")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path
