"""Parsers that convert dataset-specific annotation formats into the unified
`ImageAnnotation` / `BoundingBox` schema.

Supports:
    - YOLO txt          (class_id x_center y_center width height, normalized)
    - Pascal VOC XML     (<annotation><object><bndbox> ...)
    - COCO JSON          (single json file, images/annotations/categories)
    - Generic CSV        (one row per box; column names configurable)

Kaggle license-plate datasets are inconsistent in practice, so
`detect_annotation_format` performs light-weight sniffing and
`parse_annotation_file` dispatches accordingly. Unknown / unparsable
files are skipped with a warning rather than crashing the pipeline.
"""

from __future__ import annotations

import csv
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from alpr_dataset.annotations.schema import BoundingBox, ImageAnnotation
from alpr_dataset.io_utils import build_image_meta

logger = logging.getLogger("alpr_dataset")

DEFAULT_CLASS_NAME = "license_plate"
DEFAULT_CLASS_ID = 0


def detect_annotation_format(path: Path) -> str:
    """Best-effort sniff of an annotation file's format from extension + content."""
    suffix = path.suffix.lower()
    if suffix == ".xml":
        return "voc_xml"
    if suffix == ".json":
        return "coco_json"
    if suffix == ".csv":
        return "csv"
    if suffix == ".txt":
        return "yolo"
    return "unknown"


def parse_yolo(
    annotation_path: Path,
    image_path: Path,
    dataset_name: str,
    class_map: dict[int, str] | None = None,
) -> ImageAnnotation | None:
    meta = build_image_meta(image_path, dataset_name)
    if meta is None:
        logger.warning("Skipping unreadable image for YOLO annotation: %s", image_path)
        return None

    class_map = class_map or {}
    boxes: list[BoundingBox] = []
    try:
        for line in annotation_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                logger.warning("Malformed YOLO line in %s: '%s'", annotation_path, line)
                continue
            class_id = int(float(parts[0]))
            cx, cy, w, h = (float(x) for x in parts[1:5])
            x_min = (cx - w / 2) * meta.width
            y_min = (cy - h / 2) * meta.height
            x_max = (cx + w / 2) * meta.width
            y_max = (cy + h / 2) * meta.height
            boxes.append(
                BoundingBox(
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    class_id=class_id,
                    class_name=class_map.get(class_id, DEFAULT_CLASS_NAME),
                )
            )
    except OSError as exc:
        logger.warning("Could not read YOLO annotation %s: %s", annotation_path, exc)
        return None

    return ImageAnnotation(
        image_path=image_path,
        annotation_path=annotation_path,
        image_width=meta.width,
        image_height=meta.height,
        dataset_name=dataset_name,
        boxes=boxes,
        source_format="yolo",
    )


def parse_voc_xml(
    annotation_path: Path,
    image_path: Path,
    dataset_name: str,
    class_map: dict[int, str] | None = None,
) -> ImageAnnotation | None:
    meta = build_image_meta(image_path, dataset_name)
    if meta is None:
        logger.warning("Skipping unreadable image for VOC annotation: %s", image_path)
        return None

    name_to_id: dict[str, int] = {}
    if class_map:
        name_to_id = {v: k for k, v in class_map.items()}

    boxes: list[BoundingBox] = []
    try:
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            name_el = obj.find("name")
            class_name = name_el.text.strip() if name_el is not None and name_el.text else DEFAULT_CLASS_NAME
            class_id = name_to_id.get(class_name, DEFAULT_CLASS_ID)
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            x_min = float(bnd.findtext("xmin", default="0"))
            y_min = float(bnd.findtext("ymin", default="0"))
            x_max = float(bnd.findtext("xmax", default="0"))
            y_max = float(bnd.findtext("ymax", default="0"))
            boxes.append(
                BoundingBox(
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    class_id=class_id,
                    class_name=class_name,
                )
            )
    except (ET.ParseError, OSError) as exc:
        logger.warning("Could not parse VOC XML %s: %s", annotation_path, exc)
        return None

    return ImageAnnotation(
        image_path=image_path,
        annotation_path=annotation_path,
        image_width=meta.width,
        image_height=meta.height,
        dataset_name=dataset_name,
        boxes=boxes,
        source_format="voc_xml",
    )


def parse_coco_json(
    json_path: Path,
    images_dir: Path,
    dataset_name: str,
) -> list[ImageAnnotation]:
    """Parse a single COCO-style JSON file covering an entire dataset split."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not parse COCO JSON %s: %s", json_path, exc)
        return []

    categories = {c["id"]: c["name"] for c in data.get("categories", [])}
    images_by_id = {img["id"]: img for img in data.get("images", [])}

    boxes_by_image: dict[int, list[BoundingBox]] = {}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        x, y, w, h = ann["bbox"]  # COCO bbox: [x_min, y_min, width, height]
        class_id = ann["category_id"]
        box = BoundingBox(
            x_min=x,
            y_min=y,
            x_max=x + w,
            y_max=y + h,
            class_id=class_id,
            class_name=categories.get(class_id, DEFAULT_CLASS_NAME),
        )
        boxes_by_image.setdefault(img_id, []).append(box)

    results: list[ImageAnnotation] = []
    for img_id, img_info in images_by_id.items():
        image_path = images_dir / img_info["file_name"]
        results.append(
            ImageAnnotation(
                image_path=image_path,
                annotation_path=json_path,
                image_width=img_info.get("width", 0),
                image_height=img_info.get("height", 0),
                dataset_name=dataset_name,
                boxes=boxes_by_image.get(img_id, []),
                source_format="coco_json",
            )
        )
    return results


def parse_csv_annotations(
    csv_path: Path,
    images_dir: Path,
    dataset_name: str,
    filename_col: str = "filename",
    box_cols: tuple[str, str, str, str] = ("xmin", "ymin", "xmax", "ymax"),
    class_col: str | None = "class",
) -> list[ImageAnnotation]:
    """Parse a generic CSV with one row per bounding box.

    Many Kaggle plate-detection datasets ship a single CSV of this shape
    instead of per-image txt/xml files.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.warning("Could not parse annotation CSV %s: %s", csv_path, exc)
        return []

    missing_cols = [c for c in (filename_col, *box_cols) if c not in df.columns]
    if missing_cols:
        logger.warning(
            "CSV %s missing expected columns %s (found: %s)",
            csv_path,
            missing_cols,
            list(df.columns),
        )
        return []

    grouped: dict[str, list[BoundingBox]] = {}
    widths_heights: dict[str, tuple[int, int]] = {}
    for _, row in df.iterrows():
        fname = str(row[filename_col])
        class_name = str(row[class_col]) if class_col and class_col in df.columns else DEFAULT_CLASS_NAME
        box = BoundingBox(
            x_min=float(row[box_cols[0]]),
            y_min=float(row[box_cols[1]]),
            x_max=float(row[box_cols[2]]),
            y_max=float(row[box_cols[3]]),
            class_id=DEFAULT_CLASS_ID,
            class_name=class_name,
        )
        grouped.setdefault(fname, []).append(box)
        if "width" in df.columns and "height" in df.columns:
            widths_heights[fname] = (int(row["width"]), int(row["height"]))

    results: list[ImageAnnotation] = []
    for fname, boxes in grouped.items():
        image_path = images_dir / fname
        if fname in widths_heights:
            w, h = widths_heights[fname]
        else:
            meta = build_image_meta(image_path, dataset_name)
            w, h = (meta.width, meta.height) if meta else (0, 0)
        results.append(
            ImageAnnotation(
                image_path=image_path,
                annotation_path=csv_path,
                image_width=w,
                image_height=h,
                dataset_name=dataset_name,
                boxes=boxes,
                source_format="csv",
            )
        )
    return results


def parse_annotation_file(
    annotation_path: Path,
    image_path: Path,
    dataset_name: str,
    annotation_format: str | None = None,
    class_map: dict[int, str] | None = None,
) -> ImageAnnotation | None:
    """Single-file dispatcher for the per-image formats (YOLO / VOC XML).

    COCO JSON and CSV formats are dataset-wide, not per-image — use
    `parse_coco_json` / `parse_csv_annotations` directly for those.
    """
    fmt = annotation_format or detect_annotation_format(annotation_path)
    if fmt == "yolo":
        return parse_yolo(annotation_path, image_path, dataset_name, class_map)
    if fmt == "voc_xml":
        return parse_voc_xml(annotation_path, image_path, dataset_name, class_map)
    logger.warning("Unsupported per-image annotation format '%s' for %s", fmt, annotation_path)
    return None
