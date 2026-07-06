"""Unified annotation schema.

Every dataset-specific parser (YOLO txt, Pascal VOC XML, COCO JSON, custom
CSV) converts into this common representation so downstream EDA, quality
checks, harmonization and preprocessing never need to know the original
source format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BoundingBox:
    """Absolute pixel-space bounding box, top-left origin."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    class_id: int
    class_name: str = ""
    plate_text: str | None = None  # ground-truth plate text, if available

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2

    def to_normalized(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        """Return (x_center, y_center, width, height), all normalized 0-1 (YOLO style)."""
        cx, cy = self.center
        return cx / img_w, cy / img_h, self.width / img_w, self.height / img_h

    def iou(self, other: "BoundingBox") -> float:
        ix_min = max(self.x_min, other.x_min)
        iy_min = max(self.y_min, other.y_min)
        ix_max = min(self.x_max, other.x_max)
        iy_max = min(self.y_max, other.y_max)
        iw = max(0.0, ix_max - ix_min)
        ih = max(0.0, iy_max - iy_min)
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass
class ImageAnnotation:
    """All annotations belonging to a single image."""

    image_path: Path
    annotation_path: Path | None
    image_width: int
    image_height: int
    dataset_name: str
    boxes: list[BoundingBox] = field(default_factory=list)
    source_format: str = "unknown"

    @property
    def n_boxes(self) -> int:
        return len(self.boxes)
