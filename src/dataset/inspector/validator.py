from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from src.dataset.inspector.scanner import ScanResult
from src.dataset.utils.filesystem import AnnotationFormat
from src.dataset.utils.hashing import find_exact_duplicates, find_near_duplicates
from src.dataset.utils.image import has_supported_extension, is_corrupted, is_readable, is_zero_byte


@dataclass
class ValidationIssue:
    issue_type: str
    file_path: str
    message: str
    details: dict | None = None


@dataclass
class ValidationSummary:
    total_images: int = 0
    total_annotations: int = 0
    corrupted_images: list[ValidationIssue] = field(default_factory=list)
    unreadable_images: list[ValidationIssue] = field(default_factory=list)
    zero_byte_files: list[ValidationIssue] = field(default_factory=list)
    unsupported_extensions: list[ValidationIssue] = field(default_factory=list)
    missing_labels: list[ValidationIssue] = field(default_factory=list)
    orphan_labels: list[ValidationIssue] = field(default_factory=list)
    empty_annotations: list[ValidationIssue] = field(default_factory=list)
    invalid_bboxes: list[ValidationIssue] = field(default_factory=list)
    duplicate_filenames: list[ValidationIssue] = field(default_factory=list)
    duplicate_images: list[ValidationIssue] = field(default_factory=list)
    near_duplicates: list[ValidationIssue] = field(default_factory=list)

    @property
    def total_issues(self) -> int:
        return (
            len(self.corrupted_images)
            + len(self.unreadable_images)
            + len(self.zero_byte_files)
            + len(self.unsupported_extensions)
            + len(self.missing_labels)
            + len(self.orphan_labels)
            + len(self.empty_annotations)
            + len(self.invalid_bboxes)
            + len(self.duplicate_filenames)
            + len(self.duplicate_images)
            + len(self.near_duplicates)
        )

    def to_dict(self) -> dict:
        return {
            "total_images": self.total_images,
            "total_annotations": self.total_annotations,
            "total_issues": self.total_issues,
            "corrupted_images": [
                {"file": i.file_path, "message": i.message, "details": i.details} for i in self.corrupted_images
            ],
            "unreadable_images": [
                {"file": i.file_path, "message": i.message, "details": i.details} for i in self.unreadable_images
            ],
            "zero_byte_files": [
                {"file": i.file_path, "message": i.message, "details": i.details} for i in self.zero_byte_files
            ],
            "unsupported_extensions": [
                {"file": i.file_path, "message": i.message, "details": i.details} for i in self.unsupported_extensions
            ],
            "missing_labels": [
                {"file": i.file_path, "message": i.message, "details": i.details} for i in self.missing_labels
            ],
            "orphan_labels": [
                {"file": i.file_path, "message": i.message, "details": i.details} for i in self.orphan_labels
            ],
            "empty_annotations": [
                {"file": i.file_path, "message": i.message, "details": i.details} for i in self.empty_annotations
            ],
            "invalid_bboxes": [
                {"file": i.file_path, "message": i.message, "details": i.details} for i in self.invalid_bboxes
            ],
            "duplicate_filenames": [
                {"file": i.file_path, "message": i.message, "details": i.details} for i in self.duplicate_filenames
            ],
            "duplicate_images": [
                {"file": i.file_path, "message": i.message, "details": i.details} for i in self.duplicate_images
            ],
            "near_duplicates": [
                {"file": i.file_path, "message": i.message, "details": i.details} for i in self.near_duplicates
            ],
        }


def _validate_voc_bbox(xml_path: Path) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            if bndbox is None:
                issues.append(ValidationIssue(
                    issue_type="invalid_bbox",
                    file_path=str(xml_path),
                    message="Missing bndbox element",
                ))
                continue
            try:
                xmin = int(bndbox.findtext("xmin", "0"))
                ymin = int(bndbox.findtext("ymin", "0"))
                xmax = int(bndbox.findtext("xmax", "0"))
                ymax = int(bndbox.findtext("ymax", "0"))
            except (ValueError, TypeError):
                issues.append(ValidationIssue(
                    issue_type="invalid_bbox",
                    file_path=str(xml_path),
                    message="Non-integer bbox coordinates",
                ))
                continue
            if xmin >= xmax or ymin >= ymax:
                issues.append(ValidationIssue(
                    issue_type="invalid_bbox",
                    file_path=str(xml_path),
                    message=f"Invalid bbox dimensions: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}",
                    details={"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
                ))
            if xmin < 0 or ymin < 0:
                issues.append(ValidationIssue(
                    issue_type="invalid_bbox",
                    file_path=str(xml_path),
                    message=f"Negative bbox coordinates: xmin={xmin}, ymin={ymin}",
                    details={"xmin": xmin, "ymin": ymin},
                ))
            size_elem = root.find("size")
            if size_elem is not None:
                width = int(size_elem.findtext("width", "0"))
                height = int(size_elem.findtext("height", "0"))
                if width > 0 and height > 0:
                    if xmax > width or ymax > height:
                        issues.append(ValidationIssue(
                            issue_type="invalid_bbox",
                            file_path=str(xml_path),
                            message=f"Bbox exceeds image dimensions: ({xmax},{ymax}) > ({width},{height})",
                            details={"xmax": xmax, "ymax": ymax, "img_width": width, "img_height": height},
                        ))
    except ET.ParseError as e:
        issues.append(ValidationIssue(
            issue_type="invalid_bbox",
            file_path=str(xml_path),
            message=f"XML parse error: {e}",
        ))
    return issues


def _validate_yolo_bbox(txt_path: Path) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    try:
        content = txt_path.read_text().strip()
        if not content:
            return issues
        for line_num, line in enumerate(content.split("\n"), 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                issues.append(ValidationIssue(
                    issue_type="invalid_bbox",
                    file_path=str(txt_path),
                    message=f"Line {line_num}: expected 5 values, got {len(parts)}",
                    details={"line": line_num, "values": parts},
                ))
                continue
            try:
                vals = [float(v) for v in parts[1:]]
            except ValueError:
                issues.append(ValidationIssue(
                    issue_type="invalid_bbox",
                    file_path=str(txt_path),
                    message=f"Line {line_num}: non-numeric values",
                    details={"line": line_num, "content": line},
                ))
                continue
            cx, cy, w, h = vals
            if w <= 0 or h <= 0:
                issues.append(ValidationIssue(
                    issue_type="invalid_bbox",
                    file_path=str(txt_path),
                    message=f"Line {line_num}: non-positive width/height: w={w}, h={h}",
                    details={"line": line_num, "width": w, "height": h},
                ))
            if cx < 0 or cx > 1 or cy < 0 or cy > 1:
                issues.append(ValidationIssue(
                    issue_type="invalid_bbox",
                    file_path=str(txt_path),
                    message=f"Line {line_num}: normalized coords out of [0,1]: cx={cx}, cy={cy}",
                    details={"line": line_num, "cx": cx, "cy": cy},
                ))
    except Exception as e:
        issues.append(ValidationIssue(
            issue_type="invalid_bbox",
            file_path=str(txt_path),
            message=f"Read error: {e}",
        ))
    return issues


def _check_empty_annotation(ann_path: Path, ann_format: AnnotationFormat) -> bool:
    try:
        if ann_format == AnnotationFormat.VOC_XML:
            tree = ET.parse(ann_path)
            root = tree.getroot()
            objects = root.findall("object")
            return len(objects) == 0
        if ann_format == AnnotationFormat.YOLO_TXT:
            content = ann_path.read_text().strip()
            return len(content) == 0
        if ann_format == AnnotationFormat.COCO_JSON:
            content = ann_path.read_text().strip()
            return len(content) == 0
    except Exception:
        return False
    return False


class DatasetValidator:
    def __init__(
        self,
        use_tqdm: bool = True,
        near_duplicate_threshold: int = 8,
    ) -> None:
        self.use_tqdm = use_tqdm
        self.near_duplicate_threshold = near_duplicate_threshold

    def validate(self, scan_result: ScanResult) -> ValidationSummary:
        summary = ValidationSummary(
            total_images=scan_result.total_images,
            total_annotations=scan_result.total_annotations,
        )

        summary.corrupted_images = self._check_corrupted(scan_result.image_files)
        summary.unreadable_images = self._check_unreadable(scan_result.image_files)
        summary.zero_byte_files = self._check_zero_byte(scan_result.image_files + scan_result.annotation_files)
        summary.unsupported_extensions = self._check_unsupported_extensions(scan_result.image_files)
        summary.orphan_labels = self._check_orphan_labels(scan_result)
        summary.missing_labels = self._check_missing_labels(scan_result)
        summary.empty_annotations = self._check_empty_annotations(scan_result)
        summary.invalid_bboxes = self._check_invalid_bboxes(scan_result)
        summary.duplicate_filenames = self._check_duplicate_filenames(scan_result.image_files)
        summary.duplicate_images = self._check_duplicate_images(scan_result.image_files)
        summary.near_duplicates = self._check_near_duplicates(scan_result.image_files)

        return summary

    def _check_corrupted(self, image_files: list[Path]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for img in tqdm(image_files, desc="Checking corrupted images", disable=not self.use_tqdm):
            if is_corrupted(img):
                issues.append(ValidationIssue(
                    issue_type="corrupted_image",
                    file_path=str(img),
                    message="Image data is corrupted or cannot be decoded",
                ))
        return issues

    def _check_unreadable(self, image_files: list[Path]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for img in tqdm(image_files, desc="Checking readable images", disable=not self.use_tqdm):
            if not is_readable(img):
                issues.append(ValidationIssue(
                    issue_type="unreadable_image",
                    file_path=str(img),
                    message="Image file cannot be opened or verified by PIL",
                ))
        return issues

    def _check_zero_byte(self, files: list[Path]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for f in tqdm(files, desc="Checking zero-byte files", disable=not self.use_tqdm):
            if is_zero_byte(f):
                issues.append(ValidationIssue(
                    issue_type="zero_byte",
                    file_path=str(f),
                    message="File is zero bytes",
                ))
        return issues

    def _check_unsupported_extensions(self, image_files: list[Path]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for img in image_files:
            if not has_supported_extension(img):
                issues.append(ValidationIssue(
                    issue_type="unsupported_extension",
                    file_path=str(img),
                    message=f"Unsupported image extension: {img.suffix}",
                ))
        return issues

    def _check_orphan_labels(self, scan_result: ScanResult) -> list[ValidationIssue]:
        return [
            ValidationIssue(
                issue_type="orphan_label",
                file_path=str(ann),
                message="Annotation file has no matching image",
            )
            for ann in scan_result.orphan_annotations
        ]

    def _check_missing_labels(self, scan_result: ScanResult) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        image_stems = {img.stem for img in scan_result.image_files}
        ann_stems = {ann.stem for ann in scan_result.annotation_files}

        if scan_result.annotation_format == AnnotationFormat.VOC_XML:
            for ann in scan_result.annotation_files:
                try:
                    tree = ET.parse(ann)
                    root = tree.getroot()
                    filename_elem = root.find("filename")
                    if filename_elem is not None and filename_elem.text is not None:
                        ref_stem = Path(filename_elem.text).stem
                        if ref_stem not in image_stems and ann.stem not in image_stems:
                            issues.append(ValidationIssue(
                                issue_type="missing_label",
                                file_path=str(ann),
                                message=f"Referenced image '{filename_elem.text}' not found in dataset",
                            ))
                except ET.ParseError:
                    pass

        for img_stem in image_stems:
            if img_stem not in ann_stems:
                issues.append(ValidationIssue(
                    issue_type="missing_label",
                    file_path=str([str(p) for p in scan_result.image_files if p.stem == img_stem][0]),
                    message="Image has no corresponding annotation file",
                ))
        return issues

    def _check_empty_annotations(self, scan_result: ScanResult) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for ann in tqdm(scan_result.annotation_files, desc="Checking empty annotations", disable=not self.use_tqdm):
            if _check_empty_annotation(ann, scan_result.annotation_format):
                issues.append(ValidationIssue(
                    issue_type="empty_annotation",
                    file_path=str(ann),
                    message="Annotation file contains no objects/labels",
                ))
        return issues

    def _check_invalid_bboxes(self, scan_result: ScanResult) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        ann_format = scan_result.annotation_format
        for ann in tqdm(scan_result.annotation_files, desc="Checking bboxes", disable=not self.use_tqdm):
            if ann_format == AnnotationFormat.VOC_XML:
                issues.extend(_validate_voc_bbox(ann))
            elif ann_format == AnnotationFormat.YOLO_TXT:
                issues.extend(_validate_yolo_bbox(ann))
        return issues

    def _check_duplicate_filenames(self, image_files: list[Path]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        name_map: dict[str, list[Path]] = {}
        for img in image_files:
            name_map.setdefault(img.name, []).append(img)
        for name, paths in name_map.items():
            if len(paths) > 1:
                issues.append(ValidationIssue(
                    issue_type="duplicate_filename",
                    file_path=str(paths[0]),
                    message=f"Duplicate filename '{name}' appears {len(paths)} times",
                    details={"filename": name, "count": len(paths), "paths": [str(p) for p in paths]},
                ))
        return issues

    def _check_duplicate_images(self, image_files: list[Path]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        dup_map = find_exact_duplicates(image_files)
        for md5, paths in dup_map.items():
            issues.append(ValidationIssue(
                issue_type="duplicate_image",
                file_path=str(paths[0]),
                message=f"Exact duplicate (MD5: {md5}) found {len(paths)} times",
                details={"md5": md5, "count": len(paths), "paths": [str(p) for p in paths]},
            ))
        return issues

    def _check_near_duplicates(self, image_files: list[Path]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        pairs = find_near_duplicates(image_files, threshold=self.near_duplicate_threshold)
        for p1, p2, diff in pairs:
            issues.append(ValidationIssue(
                issue_type="near_duplicate",
                file_path=str(p1),
                message=f"Near-duplicate of {p2.name} (Hamming distance: {diff})",
                details={"file1": str(p1), "file2": str(p2), "hamming_distance": diff},
            ))
        return issues
