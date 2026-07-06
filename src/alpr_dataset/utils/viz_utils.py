"""Small shared visualization helpers (color conversion, bbox drawing)."""

from __future__ import annotations

import cv2
import numpy as np

from alpr_dataset.annotations.schema import BoundingBox


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)


def draw_boxes(image: np.ndarray, boxes: list[BoundingBox], color: tuple[int, int, int] = (0, 0, 255), thickness: int = 2) -> np.ndarray:
    """Draw bounding boxes on a copy of the image (BGR, OpenCV convention)."""
    out = image.copy()
    for box in boxes:
        cv2.rectangle(
            out,
            (int(box.x_min), int(box.y_min)),
            (int(box.x_max), int(box.y_max)),
            color,
            thickness,
        )
        if box.class_name:
            cv2.putText(
                out,
                box.class_name,
                (int(box.x_min), max(0, int(box.y_min) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    return out
