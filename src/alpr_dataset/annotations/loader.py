"""High-level loader: given a DatasetSpec, produce a list[ImageAnnotation].

This is the single integration point that every downstream module (EDA,
quality, harmonization, preprocessing, statistics, splitting) uses to get
a uniform view of a dataset, regardless of its original annotation format.
"""

from __future__ import annotations

import logging

from alpr_dataset.annotations.parsers import (
    parse_annotation_file,
    parse_coco_json,
    parse_csv_annotations,
)
from alpr_dataset.annotations.schema import ImageAnnotation
from alpr_dataset.config import DatasetSpec
from alpr_dataset.io_utils import list_annotations, list_images

logger = logging.getLogger("alpr_dataset")


def load_dataset_annotations(spec: DatasetSpec) -> list[ImageAnnotation]:
    """Load all per-image annotations for a configured dataset.

    Dispatches on `spec.annotation_format`:
      - "yolo" / "voc_xml": one annotation file per image, matched by stem.
      - "coco_json": a single JSON file somewhere under the dataset root.
      - "csv": a single CSV file somewhere under the dataset root.
    """
    images_root = spec.root / spec.image_subdir if spec.image_subdir else spec.root
    annotations_root = (
        spec.root / spec.annotation_subdir if spec.annotation_subdir else spec.root
    )

    images = list_images(images_root)

    if spec.annotation_format in ("yolo", "voc_xml"):
        annotations = list_annotations(annotations_root)
        by_stem = {p.stem: p for p in annotations}
        results: list[ImageAnnotation] = []
        for img_path in images:
            ann_path = by_stem.get(img_path.stem)
            if ann_path is None:
                logger.debug("No annotation found for image %s", img_path)
                continue
            parsed = parse_annotation_file(
                ann_path, img_path, spec.name, spec.annotation_format, spec.class_map
            )
            if parsed is not None:
                results.append(parsed)
        return results

    if spec.annotation_format == "coco_json":
        json_candidates = list(annotations_root.rglob("*.json"))
        if not json_candidates:
            logger.warning("No COCO JSON file found under %s", annotations_root)
            return []
        return parse_coco_json(json_candidates[0], images_root, spec.name)

    if spec.annotation_format == "csv":
        csv_candidates = list(annotations_root.rglob("*.csv"))
        if not csv_candidates:
            logger.warning("No CSV annotation file found under %s", annotations_root)
            return []
        return parse_csv_annotations(csv_candidates[0], images_root, spec.name)

    logger.warning(
        "Unknown annotation_format '%s' for dataset '%s' — returning image-only "
        "records with zero boxes.",
        spec.annotation_format,
        spec.name,
    )
    from alpr_dataset.io_utils import build_image_meta

    results = []
    for img_path in images:
        meta = build_image_meta(img_path, spec.name)
        if meta is None:
            continue
        results.append(
            ImageAnnotation(
                image_path=img_path,
                annotation_path=None,
                image_width=meta.width,
                image_height=meta.height,
                dataset_name=spec.name,
                boxes=[],
                source_format="unknown",
            )
        )
    return results
