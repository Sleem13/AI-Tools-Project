from __future__ import annotations

from pathlib import Path

import cv2
from PIL import Image

from src.dataset.utils.filesystem import SUPPORTED_IMAGE_EXTENSIONS


def is_zero_byte(file_path: Path) -> bool:
    return file_path.stat().st_size == 0


def has_supported_extension(file_path: Path) -> bool:
    return file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def is_readable(file_path: Path) -> bool:
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def is_corrupted(file_path: Path) -> bool:
    try:
        img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return True
        if img.size == 0:
            return True
        return False
    except Exception:
        return True


def get_image_size_pil(file_path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(file_path) as img:
            return img.size
    except Exception:
        return None


def get_image_size_cv(file_path: Path) -> tuple[int, int] | None:
    try:
        img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        return (img.shape[1], img.shape[0])
    except Exception:
        return None


def get_image_mode(file_path: Path) -> str | None:
    try:
        with Image.open(file_path) as img:
            return img.mode
    except Exception:
        return None


def validate_image(file_path: Path) -> dict[str, bool | str | tuple[int, int] | None]:
    result: dict[str, bool | str | tuple[int, int] | None] = {
        "path": str(file_path),
        "exists": file_path.exists(),
        "zero_byte": False,
        "supported_extension": True,
        "readable": True,
        "corrupted": False,
        "size": None,
        "mode": None,
        "error": None,
    }

    if not file_path.exists():
        result["error"] = "File does not exist"
        return result

    result["zero_byte"] = is_zero_byte(file_path)
    if result["zero_byte"]:
        result["error"] = "Zero-byte file"
        return result

    result["supported_extension"] = has_supported_extension(file_path)
    if not result["supported_extension"]:
        result["error"] = "Unsupported extension"

    result["readable"] = is_readable(file_path)
    if not result["readable"]:
        result["error"] = "Unreadable image file"
        return result

    result["corrupted"] = is_corrupted(file_path)
    if result["corrupted"]:
        result["error"] = "Corrupted image data"

    result["size"] = get_image_size_pil(file_path)
    result["mode"] = get_image_mode(file_path)

    return result
