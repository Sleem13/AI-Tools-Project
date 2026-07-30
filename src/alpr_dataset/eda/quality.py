"""Part 3 — Data quality assessment.

Aggregates duplicate detection, corrupted-file detection, and annotation
validity checks into a single quality report (JSON + Markdown).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from tqdm import tqdm

from alpr_dataset.annotations.schema import ImageAnnotation
from alpr_dataset.annotations.validators import (
    is_empty_image_annotation,
    validate_image_annotation,
)
from alpr_dataset.inspection.hashing import find_duplicates
from alpr_dataset.inspection.image_stats import compute_image_stats
from alpr_dataset.io_utils import ensure_dir

logger = logging.getLogger("alpr_dataset")


def build_quality_report(
    dataset_name: str,
    annotations: list[ImageAnnotation],
    image_paths: list[Path],
    output_dir: Path,
    valid_class_ids: set[int] | None = None,
    hamming_threshold: int = 5,
    overlap_iou_threshold: float = 0.7,
    blur_threshold: float = 100.0,
) -> dict:
    """Run all Part-3 checks for one dataset and persist a JSON + MD report.

    Args:
        dataset_name: Name of the dataset (used in output filenames).
        annotations: Unified ImageAnnotation records for this dataset.
        image_paths: All image paths scanned for this dataset (superset of
                     images that actually have annotations).
        output_dir: Directory to write the quality report into.
        valid_class_ids: Set of class ids considered valid; anything else is
                          flagged as an "incorrect label id".
        hamming_threshold: Perceptual hash distance threshold for near-dupes.
        overlap_iou_threshold: IoU above which two boxes count as "overlapping".
        blur_threshold: Variance-of-Laplacian value below which an image is flagged as blurry.
    """
    output_dir = ensure_dir(output_dir)

    # Corrupted / empty / blurry images
    corrupted_images: list[str] = []
    empty_images: list[str] = []
    blurry_images: list[str] = []
    for p in tqdm(image_paths, desc=f"[{dataset_name}] quality: images", leave=False):
        s = compute_image_stats(p)
        if s.is_corrupted:
            corrupted_images.append(str(p))
        elif s.brightness_mean == 0 and s.contrast_std == 0:
            empty_images.append(str(p))
        if not s.is_corrupted and s.blur_score < blur_threshold:
            blurry_images.append(str(p))

    # Duplicates
    dupe_groups = find_duplicates(image_paths, hamming_threshold=hamming_threshold)

    # Annotation issues (missing / orphan handled upstream by scanner; this
    # module focuses on structural bbox validity + empty annotation files).
    annotated_image_stems = {a.image_path.stem for a in annotations}
    missing_labels = [
        str(p) for p in image_paths if p.stem not in annotated_image_stems
    ]

    boxes_outside_total = 0
    zero_area_total = 0
    overlapping_total = 0
    incorrect_label_ids: list[int] = []
    empty_annotation_files: list[str] = []

    for ann in tqdm(annotations, desc=f"[{dataset_name}] quality: annotations", leave=False):
        issues = validate_image_annotation(
            ann, valid_class_ids=valid_class_ids, overlap_iou_threshold=overlap_iou_threshold
        )
        boxes_outside_total += issues.boxes_outside_image
        zero_area_total += issues.zero_area_boxes
        overlapping_total += issues.overlapping_box_pairs
        incorrect_label_ids.extend(issues.unknown_class_ids)
        if is_empty_image_annotation(ann):
            empty_annotation_files.append(str(ann.annotation_path))

    orphan_labels = [
        str(a.annotation_path)
        for a in annotations
        if a.annotation_path is not None and not Path(a.image_path).exists()
    ]

    report = {
        "dataset_name": dataset_name,
        "n_images_scanned": len(image_paths),
        "n_corrupted_images": len(corrupted_images),
        "corrupted_images": corrupted_images[:200],
        "n_empty_images": len(empty_images),
        "empty_images": empty_images[:200],
        "n_exact_duplicate_groups": len(dupe_groups.exact_duplicates),
        "exact_duplicate_groups": {
            k: [str(p) for p in v] for k, v in list(dupe_groups.exact_duplicates.items())[:100]
        },
        "n_near_duplicate_pairs": len(dupe_groups.near_duplicates),
        "near_duplicate_pairs": [
            {"a": str(a), "b": str(b), "hamming_distance": d}
            for a, b, d in dupe_groups.near_duplicates[:200]
        ],
        "n_missing_labels": len(missing_labels),
        "missing_labels": missing_labels[:200],
        "n_orphan_labels": len(orphan_labels),
        "orphan_labels": orphan_labels[:200],
        "n_empty_annotation_files": len(empty_annotation_files),
        "empty_annotation_files": empty_annotation_files[:200],
        "n_boxes_outside_image": boxes_outside_total,
        "n_zero_area_boxes": zero_area_total,
        "n_overlapping_box_pairs": overlapping_total,
        "n_incorrect_label_ids": len(incorrect_label_ids),
        "incorrect_label_ids_sample": sorted(set(incorrect_label_ids))[:50],
        "n_blurry_images": len(blurry_images),
        "blurry_images": blurry_images[:200],
    }

    json_path = output_dir / f"{dataset_name}_quality_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote quality report JSON -> %s", json_path)

    md_path = output_dir / f"{dataset_name}_quality_report.md"
    _write_quality_markdown(report, md_path)
    logger.info("Wrote quality report Markdown -> %s", md_path)

    return report


def _write_quality_markdown(report: dict, md_path: Path) -> None:
    lines = [f"# Data Quality Report — {report['dataset_name']}", ""]
    lines.append(f"Images scanned: **{report['n_images_scanned']}**")
    lines.append("")
    lines.append("| Check | Count |")
    lines.append("|---|---|")
    checks = [
        ("Corrupted images", report["n_corrupted_images"]),
        ("Empty images", report["n_empty_images"]),
        ("Exact duplicate groups", report["n_exact_duplicate_groups"]),
        ("Near-duplicate pairs", report["n_near_duplicate_pairs"]),
        ("Missing labels (image w/o annotation)", report["n_missing_labels"]),
        ("Orphan labels (annotation w/o image)", report["n_orphan_labels"]),
        ("Empty annotation files", report["n_empty_annotation_files"]),
        ("Boxes outside image bounds", report["n_boxes_outside_image"]),
        ("Zero-area boxes", report["n_zero_area_boxes"]),
        ("Overlapping box pairs (IoU-based)", report["n_overlapping_box_pairs"]),
        ("Incorrect / unknown label IDs", report["n_incorrect_label_ids"]),
    ]
    for name, count in checks:
        lines.append(f"| {name} | {count} |")
    lines.append("")
    lines.append(
        "> Full file-level listings (capped at 200 entries per category) are "
        "available in the companion JSON report."
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
