from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from ..utils.hashing import compute_md5, compute_phash
from ..utils.image import (
    get_image_mode,
    get_image_size_pil,
    is_corrupted,
    is_readable,
)
from ..utils.metrics import (
    compute_aspect_ratio,
    compute_blur_score,
    compute_brightness,
    compute_channels,
    compute_contrast,
    compute_entropy,
    compute_resolution,
    compute_sharpness,
    load_image_bgr,
)


@dataclass
class ImageMetadata:
    path: Path
    filename: str
    stem: str
    extension: str
    size_bytes: int
    width: int | None = None
    height: int | None = None
    channels: int | None = None
    aspect_ratio: float | None = None
    resolution: int | None = None
    mode: str | None = None
    brightness: float | None = None
    contrast: float | None = None
    sharpness: float | None = None
    blur_score: float | None = None
    entropy: float | None = None
    md5: str | None = None
    phash: str | None = None
    readable: bool = True
    corrupted: bool = False
    zero_byte: bool = False
    error: str | None = None


class MetadataExtractor:
    def __init__(
        self,
        compute_md5_flag: bool = True,
        compute_phash_flag: bool = True,
        compute_image_metrics: bool = True,
        use_tqdm: bool = True,
    ) -> None:
        self.compute_md5_flag = compute_md5_flag
        self.compute_phash_flag = compute_phash_flag
        self.compute_image_metrics = compute_image_metrics
        self.use_tqdm = use_tqdm

    def extract(self, file_path: Path) -> ImageMetadata:
        stat = file_path.stat()
        metadata = ImageMetadata(
            path=file_path,
            filename=file_path.name,
            stem=file_path.stem,
            extension=file_path.suffix.lower(),
            size_bytes=stat.st_size,
        )

        if stat.st_size == 0:
            metadata.zero_byte = True
            metadata.error = "Zero-byte file"
            return metadata

        metadata.readable = is_readable(file_path)
        if not metadata.readable:
            metadata.error = "Unreadable image file"
            return metadata

        size = get_image_size_pil(file_path)
        if size is not None:
            metadata.width, metadata.height = size

        metadata.mode = get_image_mode(file_path)

        if metadata.width is not None and metadata.height is not None:
            metadata.aspect_ratio = compute_aspect_ratio(metadata.width, metadata.height)
            metadata.resolution = compute_resolution(metadata.width, metadata.height)

        metadata.corrupted = is_corrupted(file_path)
        if metadata.corrupted:
            metadata.error = "Corrupted image data"

        if self.compute_image_metrics and not metadata.corrupted and metadata.readable:
            self._enrich_with_image_metrics(metadata)

        if self.compute_md5_flag:
            try:
                metadata.md5 = compute_md5(file_path)
            except Exception:
                pass

        if self.compute_phash_flag:
            try:
                metadata.phash = compute_phash(file_path)
            except Exception:
                pass

        return metadata

    def _enrich_with_image_metrics(self, metadata: ImageMetadata) -> None:
        img = load_image_bgr(metadata.path)
        if img is None:
            return
        try:
            metadata.channels = compute_channels(img)
            metadata.brightness = compute_brightness(img)
            metadata.contrast = compute_contrast(img)
            metadata.sharpness = compute_sharpness(img)
            metadata.blur_score = compute_blur_score(img)
            metadata.entropy = compute_entropy(img)
        except Exception:
            pass

    def extract_many(self, file_paths: list[Path]) -> list[ImageMetadata]:
        results: list[ImageMetadata] = []
        for fp in tqdm(file_paths, desc="Extracting metadata", disable=not self.use_tqdm):
            results.append(self.extract(fp))
        return results
