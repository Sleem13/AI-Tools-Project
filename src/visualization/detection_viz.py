"""Visualization helpers for detection + OCR results."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def draw_detections(
    image: np.ndarray,
    detections: list[dict],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes and labels on an image.

    Args:
        image: BGR image.
        detections: List of dicts with keys ``bbox`` (xyxy), ``confidence``, ``plate_text``.
        color: BGR tuple for box colour.
        thickness: Line thickness.

    Returns:
        Annotated BGR image.
    """
    vis = image.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.get("bbox", [0, 0, 0, 0])]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        label = det.get("plate_text", "")
        conf = det.get("confidence", 0.0)
        text = f"{label} ({conf:.2f})" if label else f"{conf:.2f}"
        cv2.putText(vis, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return vis


def draw_plate_text(
    image: np.ndarray,
    text: str,
    confidence: float,
    position: tuple[int, int] = (10, 30),
) -> np.ndarray:
    """Overlay recognised plate text at a given position."""
    vis = image.copy()
    label = f"Plate: {text} (conf={confidence:.2f})"
    cv2.putText(vis, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return vis


def create_dashboard_figure(
    original: np.ndarray,
    detections: list[dict],
    max_width: int = 800,
) -> np.ndarray:
    """Create a dashboard-ready figure annotated with detection + OCR results.

    Args:
        original: Input BGR image.
        detections: List of detection results.
        max_width: Maximum display width (resized proportionally).

    Returns:
        Annotated RGB image suitable for Streamlit / matplotlib display.
    """
    h, w = original.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w, new_h = int(w * scale), int(h * scale)
        display = cv2.resize(original, (new_w, new_h))
    else:
        display = original.copy()

    annotated = draw_detections(display, detections)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return annotated_rgb
