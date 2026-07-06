"""Shared filesystem and image I/O utilities.

Centralizing these helpers avoids duplicated file-walking / image-loading
logic across the inspection, EDA, quality, harmonization and preprocessing
modules (SOLID: single responsibility + DRY).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ANNOTATION_EXTENSIONS = {".txt", ".xml", ".json", ".csv"}


def list_files(root: Path, extensions: set[str] | None = None) -> list[Path]:
    """Recursively list files under root, optionally filtered by extension.

    Args:
        root: Directory to search.
        extensions: Set of lowercase extensions (with leading dot), e.g. {".jpg"}.
                    If None, all files are returned.

    Returns:
        Sorted list of matching file paths.
    """
    root = Path(root)
    if not root.exists():
        return []
    files = [p for p in root.rglob("*") if p.is_file()]
    if extensions:
        files = [p for p in files if p.suffix.lower() in extensions]
    return sorted(files)


def list_images(root: Path) -> list[Path]:
    return list_files(root, IMAGE_EXTENSIONS)


def list_annotations(root: Path) -> list[Path]:
    return list_files(root, ANNOTATION_EXTENSIONS)


def safe_read_image(path: Path) -> np.ndarray | None:
    """Read an image with OpenCV, returning None (not raising) on failure.

    Uses np.fromfile + cv2.imdecode to correctly support unicode / Arabic
    paths on all platforms, which cv2.imread can mishandle on Windows.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        return img
    except Exception:
        return None


def is_image_corrupted(path: Path) -> bool:
    """Return True if the image cannot be decoded or has zero-sized dimensions."""
    img = safe_read_image(path)
    if img is None:
        return True
    if img.ndim < 2 or img.shape[0] == 0 or img.shape[1] == 0:
        return True
    return False


def md5_of_file(path: Path, chunk_size: int = 65536) -> str:
    """Compute the MD5 hash of a file's raw bytes (for exact-duplicate detection)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class ImageMeta:
    """Lightweight container for per-image metadata used across modules."""

    path: Path
    width: int
    height: int
    channels: int
    file_size_bytes: int
    dataset_name: str

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0

    @property
    def resolution(self) -> int:
        return self.width * self.height


def build_image_meta(path: Path, dataset_name: str) -> ImageMeta | None:
    """Build an ImageMeta record for a single image, or None if unreadable."""
    img = safe_read_image(path)
    if img is None:
        return None
    if img.ndim == 2:
        h, w = img.shape
        c = 1
    else:
        h, w, c = img.shape
    return ImageMeta(
        path=path,
        width=w,
        height=h,
        channels=c,
        file_size_bytes=path.stat().st_size,
        dataset_name=dataset_name,
    )
