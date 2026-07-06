"""Dataset structural scanner — Part 1 of the pipeline.

Walks a dataset directory and reports folder structure, file formats,
image/annotation counts, missing annotations, and unsupported files.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from alpr_dataset.io_utils import (
    ANNOTATION_EXTENSIONS,
    IMAGE_EXTENSIONS,
    list_annotations,
    list_files,
    list_images,
)

logger = logging.getLogger("alpr_dataset")


@dataclass
class DatasetScanResult:
    dataset_name: str
    root: Path
    folder_tree: list[str] = field(default_factory=list)
    image_paths: list[Path] = field(default_factory=list)
    annotation_paths: list[Path] = field(default_factory=list)
    image_format_counts: Counter = field(default_factory=Counter)
    annotation_format_counts: Counter = field(default_factory=Counter)
    unsupported_files: list[Path] = field(default_factory=list)
    images_missing_annotations: list[Path] = field(default_factory=list)
    orphan_annotations: list[Path] = field(default_factory=list)

    @property
    def n_images(self) -> int:
        return len(self.image_paths)

    @property
    def n_annotations(self) -> int:
        return len(self.annotation_paths)


def build_folder_tree(root: Path, max_depth: int = 4) -> list[str]:
    """Produce a simple indented text representation of the folder tree."""
    lines: list[str] = []
    root = Path(root)
    if not root.exists():
        return [f"<missing: {root}>"]

    def _walk(path: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            return
        for entry in entries:
            prefix = "    " * depth + ("|-- " if depth > 0 else "")
            if entry.is_dir():
                lines.append(f"{prefix}{entry.name}/")
                _walk(entry, depth + 1)
            else:
                lines.append(f"{prefix}{entry.name}")

    lines.append(f"{root.name}/")
    _walk(root, 1)
    return lines


def scan_dataset(dataset_name: str, root: Path) -> DatasetScanResult:
    """Scan a single dataset root directory and produce a DatasetScanResult."""
    root = Path(root)
    result = DatasetScanResult(dataset_name=dataset_name, root=root)

    if not root.exists():
        logger.warning("Dataset root does not exist: %s", root)
        return result

    result.folder_tree = build_folder_tree(root)
    result.image_paths = list_images(root)
    result.annotation_paths = list_annotations(root)

    for p in result.image_paths:
        result.image_format_counts[p.suffix.lower()] += 1
    for p in result.annotation_paths:
        result.annotation_format_counts[p.suffix.lower()] += 1

    all_files = list_files(root)
    known_extensions = IMAGE_EXTENSIONS | ANNOTATION_EXTENSIONS
    result.unsupported_files = [
        p for p in all_files if p.suffix.lower() not in known_extensions
    ]

    # Match images <-> annotations by stem (filename without extension).
    annotation_stems = {p.stem: p for p in result.annotation_paths}
    image_stems = {p.stem: p for p in result.image_paths}

    result.images_missing_annotations = [
        p for p in result.image_paths if p.stem not in annotation_stems
    ]
    result.orphan_annotations = [
        p for p in result.annotation_paths if p.stem not in image_stems
    ]

    logger.info(
        "Scanned '%s': %d images, %d annotations, %d unsupported files",
        dataset_name,
        result.n_images,
        result.n_annotations,
        len(result.unsupported_files),
    )
    return result
