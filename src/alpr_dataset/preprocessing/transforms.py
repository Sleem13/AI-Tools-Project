"""Part 5 — individual, independently configurable preprocessing transforms.

Every function takes and returns a BGR (or grayscale) numpy array so they
compose freely inside `preprocessing.pipeline.PreprocessingPipeline`.
Perspective/rotation correction are provided as hooks: Egyptian plate
datasets typically already crop near-frontal plate regions, so a full
automatic corner-detector is out of scope for the dataset-prep phase, but
the hook signature lets the modeling stage plug one in later (e.g. driven
by a keypoint/segmentation model) without changing the pipeline contract.
"""

from __future__ import annotations

import cv2
import numpy as np


def resize(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize to (width, height), ignoring aspect ratio."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def letterbox(
    image: np.ndarray,
    target_size: tuple[int, int],
    color: tuple[int, int, int] = (114, 114, 114),
) -> np.ndarray:
    """Resize preserving aspect ratio, padding the remainder (YOLO-style letterbox)."""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2

    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def pad(image: np.ndarray, pad_size: int, color: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Add a uniform border of `pad_size` pixels on every side."""
    return cv2.copyMakeBorder(
        image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=color
    )


def denoise(image: np.ndarray, strength: float = 10.0) -> np.ndarray:
    """Non-local means denoising — good for the sensor noise typical of dashcam/CCTV plate crops."""
    if image.ndim == 2:
        return cv2.fastNlMeansDenoising(image, h=strength)
    return cv2.fastNlMeansDenoisingColored(image, h=strength, hColor=strength)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Global histogram equalization on the luminance channel."""
    if image.ndim == 2:
        return cv2.equalizeHist(image)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization.

    Preferred over global equalization for plates because Egyptian daytime
    captures often have strong local glare/shadow from direct sun, which
    CLAHE handles locally instead of over-amplifying global contrast.
    """
    clahe_op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    if image.ndim == 2:
        return clahe_op.apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe_op.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def gamma_correction(image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """Gamma correction — brightens under-exposed night/low-light plate crops when gamma > 1."""
    inv_gamma = 1.0 / max(gamma, 1e-6)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def brightness_normalization(image: np.ndarray, target_mean: float = 128.0) -> np.ndarray:
    """Shift pixel values so the mean intensity matches `target_mean`."""
    current_mean = image.mean()
    shift = target_mean - current_mean
    return np.clip(image.astype(np.float32) + shift, 0, 255).astype(np.uint8)


def contrast_normalization(image: np.ndarray, target_std: float = 60.0) -> np.ndarray:
    """Rescale pixel values so the intensity std. dev. matches `target_std`."""
    current_std = image.std() or 1.0
    mean = image.mean()
    scaled = (image.astype(np.float32) - mean) * (target_std / current_std) + mean
    return np.clip(scaled, 0, 255).astype(np.uint8)


def gaussian_blur(image: np.ndarray, kernel_size: int = 3, sigma: float = 0.0) -> np.ndarray:
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(image, (k, k), sigma)


def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.medianBlur(image, k)


def bilateral_filter(
    image: np.ndarray, diameter: int = 9, sigma_color: float = 75, sigma_space: float = 75
) -> np.ndarray:
    """Edge-preserving smoothing — reduces noise while keeping plate character edges crisp,
    which matters for downstream OCR legibility."""
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


def sharpen(image: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Unsharp-mask sharpening to counteract slight blur from motion / compression."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)


def perspective_correction_hook(
    image: np.ndarray, corner_points: np.ndarray | None = None
) -> np.ndarray:
    """Hook for perspective (four-point) correction of angled plate captures.

    If `corner_points` (4x2 array, TL/TR/BR/BL) is supplied — e.g. from a
    future keypoint-detection model — applies a perspective warp to a
    frontal rectangle sized to the box's own width/height. Otherwise
    returns the image unchanged, since dataset-prep does not itself run
    corner detection.
    """
    if corner_points is None or len(corner_points) != 4:
        return image
    pts_src = np.array(corner_points, dtype=np.float32)
    width = int(max(np.linalg.norm(pts_src[0] - pts_src[1]), np.linalg.norm(pts_src[3] - pts_src[2])))
    height = int(max(np.linalg.norm(pts_src[0] - pts_src[3]), np.linalg.norm(pts_src[1] - pts_src[2])))
    pts_dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    return cv2.warpPerspective(image, matrix, (width, height))


def rotation_correction_hook(image: np.ndarray, angle: float | None = None) -> np.ndarray:
    """Hook for deskewing rotated plate crops.

    If `angle` (degrees) is not supplied, estimates it via the dominant
    Hough line orientation; pass 0.0 explicitly to disable auto-estimation.
    """
    if angle is None:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=80)
        if lines is None:
            return image
        angles = [(theta * 180 / np.pi) - 90 for _rho, theta in lines[:, 0]]
        angle = float(np.median(angles))

    if abs(angle) < 0.5:
        return image
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
