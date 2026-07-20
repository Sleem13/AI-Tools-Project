from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from ..utils.filesystem import (
    AnnotationFormat,
    detect_annotation_files,
    detect_image_files,
    identify_annotation_format,
    identify_image_format,
    should_ignore,
)


@dataclass
class ScanResult:
    dataset_name: str
    root_path: Path
    image_files: list[Path] = field(default_factory=list)
    annotation_files: list[Path] = field(default_factory=list)
    annotation_format: AnnotationFormat = AnnotationFormat.UNKNOWN
    image_formats: dict[str, int] = field(default_factory=dict)
    annotation_formats: dict[str, int] = field(default_factory=dict)
    total_images: int = 0
    total_annotations: int = 0
    orphan_images: list[Path] = field(default_factory=list)
    orphan_annotations: list[Path] = field(default_factory=list)
    subdirectories: list[str] = field(default_factory=list)


def _build_stem_to_path(files: list[Path]) -> dict[str, list[Path]]:
    mapping: dict[str, list[Path]] = {}
    for f in files:
        stem = f.stem
        mapping.setdefault(stem, []).append(f)
    return mapping


class DatasetScanner:
    def __init__(
        self,
        ignore_hidden: bool = True,
        ignore_cache: bool = True,
        use_tqdm: bool = True,
    ) -> None:
        self.ignore_hidden = ignore_hidden
        self.ignore_cache = ignore_cache
        self.use_tqdm = use_tqdm

    def scan(self, dataset_path: Path) -> ScanResult:
        dataset_name = dataset_path.name

        image_files = detect_image_files(
            dataset_path,
            ignore_hidden=self.ignore_hidden,
            ignore_cache=self.ignore_cache,
        )

        annotation_files = detect_annotation_files(
            dataset_path,
            ignore_hidden=self.ignore_hidden,
            ignore_cache=self.ignore_cache,
        )

        image_format_counts: dict[str, int] = {}
        for img in image_files:
            fmt = identify_image_format(img)
            image_format_counts[fmt] = image_format_counts.get(fmt, 0) + 1

        ann_format_counts: dict[str, int] = {}
        for ann in annotation_files:
            fmt = identify_image_format(ann)
            ann_format_counts[fmt] = ann_format_counts.get(fmt, 0) + 1

        ann_format = AnnotationFormat.UNKNOWN
        if annotation_files:
            ann_format = identify_annotation_format(annotation_files[0])

        image_stems = _build_stem_to_path(image_files)
        ann_stems = _build_stem_to_path(annotation_files)

        orphan_images: list[Path] = []
        for img in tqdm(image_files, desc="Checking orphan images", disable=not self.use_tqdm):
            if img.stem not in ann_stems:
                orphan_images.append(img)

        orphan_annotations: list[Path] = []
        for ann in tqdm(annotation_files, desc="Checking orphan annotations", disable=not self.use_tqdm):
            if ann.stem not in image_stems:
                orphan_annotations.append(ann)

        subdirs: list[str] = []
        for entry in sorted(dataset_path.iterdir()):
            if entry.is_dir() and not should_ignore(entry, self.ignore_hidden, self.ignore_cache):
                subdirs.append(entry.name)

        return ScanResult(
            dataset_name=dataset_name,
            root_path=dataset_path,
            image_files=image_files,
            annotation_files=annotation_files,
            annotation_format=ann_format,
            image_formats=image_format_counts,
            annotation_formats=ann_format_counts,
            total_images=len(image_files),
            total_annotations=len(annotation_files),
            orphan_images=orphan_images,
            orphan_annotations=orphan_annotations,
            subdirectories=subdirs,
        )

    def scan_all(self, raw_path: Path) -> list[ScanResult]:
        results: list[ScanResult] = []
        if not raw_path.is_dir():
            return results
        entries = sorted(raw_path.iterdir())
        for entry in tqdm(entries, desc="Scanning datasets", disable=not self.use_tqdm):
            if not entry.is_dir():
                continue
            if should_ignore(entry, self.ignore_hidden, self.ignore_cache):
                continue
            result = self.scan(entry)
            results.append(result)
        return results
