"""Part 4 — Dataset harmonization.

Normalizes heterogeneous source datasets into a single unified layout:

    data/processed/unified/
        images/<dataset_name>__<original_stem>.jpg
        labels/<dataset_name>__<original_stem>.txt   (YOLO-normalized format)

Every transformation (rename, class remap, format conversion) is logged to
a metadata JSON file so the process is fully auditable and reproducible.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
from tqdm import tqdm

from alpr_dataset.annotations.schema import ImageAnnotation
from alpr_dataset.io_utils import ensure_dir, safe_read_image

logger = logging.getLogger("alpr_dataset")

UNIFIED_IMAGE_EXT = ".jpg"


@dataclass
class HarmonizationRecord:
    dataset_name: str
    original_image_path: str
    original_annotation_format: str
    unified_image_path: str
    unified_label_path: str
    n_boxes: int
    class_id_remap: dict[int, int] = field(default_factory=dict)


def harmonize_dataset(
    dataset_name: str,
    annotations: list[ImageAnnotation],
    unified_class_map: dict[int, str],
    local_class_map: dict[int, str],
    output_root: Path,
) -> list[HarmonizationRecord]:
    """Harmonize a single dataset's images + annotations into the unified layout.

    Args:
        dataset_name: Source dataset identifier, used as a filename prefix
                       to guarantee global uniqueness across datasets.
        annotations: Unified ImageAnnotation records for this dataset
                     (already parsed from whatever the original format was).
        unified_class_map: The pipeline-wide {class_id: class_name} target schema.
        local_class_map: This dataset's own {class_id: class_name}, used to
                          build the remap by matching class *names*.
        output_root: Root of data/processed/unified/.

    Returns:
        List of HarmonizationRecord describing every transformed file.
    """
    images_out = ensure_dir(output_root / "images")
    labels_out = ensure_dir(output_root / "labels")

    name_to_unified_id = {v: k for k, v in unified_class_map.items()}
    id_remap: dict[int, int] = {}
    for local_id, local_name in local_class_map.items():
        if local_name in name_to_unified_id:
            id_remap[local_id] = name_to_unified_id[local_name]
        else:
            logger.warning(
                "Dataset '%s': class '%s' (id=%d) has no match in unified_class_map; "
                "boxes with this class will be dropped during harmonization.",
                dataset_name,
                local_name,
                local_id,
            )

    records: list[HarmonizationRecord] = []
    for ann in tqdm(annotations, desc=f"[{dataset_name}] harmonizing", leave=False):
        img = safe_read_image(ann.image_path)
        if img is None:
            logger.warning("Skipping unreadable image during harmonization: %s", ann.image_path)
            continue

        unified_stem = f"{dataset_name}__{ann.image_path.stem}"
        unified_image_path = images_out / f"{unified_stem}{UNIFIED_IMAGE_EXT}"
        unified_label_path = labels_out / f"{unified_stem}.txt"

        # Standardize image extension / encoding.
        cv2.imwrite(str(unified_image_path), img)

        # Convert boxes to unified YOLO-normalized txt, remapping class ids
        # and dropping boxes whose class has no unified counterpart.
        lines = []
        for box in ann.boxes:
            if box.class_id not in id_remap:
                continue
            new_id = id_remap[box.class_id]
            cx, cy, w, h = box.to_normalized(ann.image_width, ann.image_height)
            lines.append(f"{new_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        unified_label_path.write_text("\n".join(lines), encoding="utf-8")

        records.append(
            HarmonizationRecord(
                dataset_name=dataset_name,
                original_image_path=str(ann.image_path),
                original_annotation_format=ann.source_format,
                unified_image_path=str(unified_image_path),
                unified_label_path=str(unified_label_path),
                n_boxes=len(lines),
                class_id_remap=id_remap,
            )
        )

    return records


def write_harmonization_metadata(records: list[HarmonizationRecord], output_dir: Path) -> Path:
    output_dir = ensure_dir(output_dir)
    path = output_dir / "harmonization_metadata.json"
    payload = {
        "n_files_harmonized": len(records),
        "unified_image_extension": UNIFIED_IMAGE_EXT,
        "unified_annotation_format": "yolo_normalized_txt",
        "records": [asdict(r) for r in records],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote harmonization metadata -> %s (%d records)", path, len(records))
    return path
