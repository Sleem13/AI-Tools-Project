from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_image_bgr(file_path: Path) -> np.ndarray | None:
    try:
        img = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        return img
    except Exception:
        return None


def compute_channels(img: np.ndarray) -> int:
    return img.shape[2] if img.ndim == 3 else 1


def compute_aspect_ratio(width: int, height: int) -> float:
    if height == 0:
        return 0.0
    return round(width / height, 6)


def compute_resolution(width: int, height: int) -> int:
    return width * height


def compute_brightness(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def compute_contrast(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray))


def compute_sharpness(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def compute_blur_score(img: np.ndarray) -> float:
    return compute_sharpness(img)


def compute_entropy(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten()
    hist_norm = hist / hist.sum()
    non_zero = hist_norm[hist_norm > 0]
    entropy_val = -float(np.sum(non_zero * np.log2(non_zero)))
    return entropy_val
