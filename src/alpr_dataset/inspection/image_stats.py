"""Per-image statistical measurements.

Computes resolution, aspect ratio, color space, file size, brightness,
contrast, blur / sharpness, and entropy for a single image. Shared by the
inspection report (Part 1), EDA figures (Part 2) and quality checks (Part 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from alpr_dataset.io_utils import safe_read_image


@dataclass
class ImageStats:
    path: Path
    width: int
    height: int
    channels: int
    color_space: str  # "grayscale", "RGB", "RGBA", "unknown"
    file_size_bytes: int
    brightness_mean: float
    contrast_std: float
    blur_score: float  # variance of Laplacian; lower = blurrier
    sharpness_score: float  # mean Sobel gradient magnitude; higher = sharper
    entropy: float
    is_corrupted: bool = False


def _shannon_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-12)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def compute_image_stats(path: Path) -> ImageStats:
    """Compute the full statistics bundle for a single image path.

    Never raises — returns is_corrupted=True with zeroed fields on failure,
    so batch pipelines can continue past broken files.
    """
    img = safe_read_image(path)
    if img is None:
        return ImageStats(
            path=path,
            width=0,
            height=0,
            channels=0,
            color_space="unknown",
            file_size_bytes=path.stat().st_size if path.exists() else 0,
            brightness_mean=0.0,
            contrast_std=0.0,
            blur_score=0.0,
            sharpness_score=0.0,
            entropy=0.0,
            is_corrupted=True,
        )

    if img.ndim == 2:
        h, w = img.shape
        c = 1
        color_space = "grayscale"
        gray = img
    else:
        h, w, c = img.shape
        color_space = {1: "grayscale", 3: "RGB", 4: "RGBA"}.get(c, "unknown")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if c >= 3 else img[:, :, 0]

    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sharpness = float(np.sqrt(sobel_x**2 + sobel_y**2).mean())

    return ImageStats(
        path=path,
        width=w,
        height=h,
        channels=c,
        color_space=color_space,
        file_size_bytes=path.stat().st_size,
        brightness_mean=float(gray.mean()),
        contrast_std=float(gray.std()),
        blur_score=laplacian_var,
        sharpness_score=sharpness,
        entropy=_shannon_entropy(gray),
        is_corrupted=False,
    )


def batch_compute_stats(paths: list[Path]) -> list[ImageStats]:
    """Compute ImageStats for a list of images. Use with tqdm at call sites."""
    return [compute_image_stats(p) for p in paths]
