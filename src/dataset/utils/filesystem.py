from __future__ import annotations

import enum
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


SUPPORTED_IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

SUPPORTED_ANNOTATION_EXTENSIONS: set[str] = {".xml", ".txt", ".json"}

IGNORE_DIRS: set[str] = {
    "__pycache__",
    ".git",
    ".svn",
    ".hg",
    ".idea",
    ".vscode",
    "__MACOSX",
    ".ipynb_checkpoints",
}

IGNORE_FILES: set[str] = {".ds_store", "thumbs.db", ".gitkeep"}


class AnnotationFormat(enum.Enum):
    VOC_XML = "voc_xml"
    YOLO_TXT = "yolo_txt"
    COCO_JSON = "coco_json"
    UNKNOWN = "unknown"


def is_hidden(path: Path) -> bool:
    parts = path.parts
    for part in parts:
        if part.startswith("."):
            return True
    return False


def is_cache_dir(path: Path) -> bool:
    return path.name in IGNORE_DIRS


def should_ignore(path: Path, ignore_hidden: bool = True, ignore_cache: bool = True) -> bool:
    name = path.name
    if name.lower() in IGNORE_FILES:
        return True
    if ignore_cache and is_cache_dir(path):
        return True
    if ignore_hidden and is_hidden(path):
        return True
    return False


def scan_files(
    root: Path,
    extensions: set[str] | None = None,
    ignore_hidden: bool = True,
    ignore_cache: bool = True,
) -> list[Path]:
    if not root.is_dir():
        return []
    files: list[Path] = []
    try:
        for entry in root.rglob("*"):
            if not entry.is_file():
                continue
            if should_ignore(entry, ignore_hidden=ignore_hidden, ignore_cache=ignore_cache):
                continue
            if should_ignore(entry.parent, ignore_hidden=ignore_hidden, ignore_cache=ignore_cache):
                continue
            if extensions is not None and entry.suffix.lower() not in extensions:
                continue
            files.append(entry)
    except PermissionError:
        pass
    return sorted(files)


def detect_image_files(
    directory: Path,
    ignore_hidden: bool = True,
    ignore_cache: bool = True,
) -> list[Path]:
    return scan_files(
        directory,
        extensions=SUPPORTED_IMAGE_EXTENSIONS,
        ignore_hidden=ignore_hidden,
        ignore_cache=ignore_cache,
    )


def detect_annotation_files(
    directory: Path,
    ignore_hidden: bool = True,
    ignore_cache: bool = True,
) -> list[Path]:
    return scan_files(
        directory,
        extensions=SUPPORTED_ANNOTATION_EXTENSIONS,
        ignore_hidden=ignore_hidden,
        ignore_cache=ignore_cache,
    )


def identify_annotation_format(file_path: Path) -> AnnotationFormat:
    suffix = file_path.suffix.lower()
    if suffix == ".xml":
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            if root.tag == "annotation":
                return AnnotationFormat.VOC_XML
        except (ET.ParseError, Exception):
            pass
        return AnnotationFormat.VOC_XML
    if suffix == ".txt":
        try:
            content = file_path.read_text().strip()
            if content and not content.startswith("{"):
                lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
                if lines:
                    parts = lines[0].split()
                    if len(parts) == 5:
                        try:
                            float(parts[1])
                            float(parts[2])
                            return AnnotationFormat.YOLO_TXT
                        except ValueError:
                            pass
        except Exception:
            pass
        return AnnotationFormat.YOLO_TXT
    if suffix == ".json":
        return AnnotationFormat.COCO_JSON
    return AnnotationFormat.UNKNOWN


def identify_image_format(file_path: Path) -> str:
    return file_path.suffix.lower().lstrip(".")


def group_files_by_dataset(raw_path: Path) -> dict[str, dict[str, Any]]:
    datasets: dict[str, dict[str, Any]] = {}
    if not raw_path.is_dir():
        return datasets
    for entry in sorted(raw_path.iterdir()):
        if not entry.is_dir():
            continue
        if should_ignore(entry):
            continue
        images = detect_image_files(entry)
        annotations = detect_annotation_files(entry)
        ann_format = AnnotationFormat.UNKNOWN
        if annotations:
            ann_format = identify_annotation_format(annotations[0])
        datasets[entry.name] = {
            "path": entry,
            "images": images,
            "annotations": annotations,
            "annotation_format": ann_format,
        }
    return datasets
