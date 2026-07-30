"""Shared fixtures for the ALPR test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure src/ is on the path so packages are importable without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture()
def tmp_image(tmp_path: Path) -> Path:
    """Create a small valid JPEG image and return its path."""
    import cv2
    import numpy as np

    img = np.zeros((64, 128, 3), dtype=np.uint8)
    img[:] = (100, 150, 200)
    path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture()
def tmp_yolo_annotation(tmp_path: Path) -> Path:
    """Create a minimal YOLO-format annotation file."""
    path = tmp_path / "test_image.txt"
    path.write_text("0 0.5 0.5 0.4 0.3\n", encoding="utf-8")
    return path


@pytest.fixture()
def tmp_voc_xml(tmp_path: Path) -> Path:
    """Create a minimal Pascal VOC XML annotation file."""
    xml = """\
<annotation>
  <filename>test_image.jpg</filename>
  <size>
    <width>128</width>
    <height>64</height>
  </size>
  <object>
    <name>license_plate</name>
    <bndbox>
      <xmin>10</xmin>
      <ymin>10</ymin>
      <xmax>90</xmax>
      <ymax>50</ymax>
    </bndbox>
  </object>
</annotation>
"""
    path = tmp_path / "test_image.xml"
    path.write_text(xml, encoding="utf-8")
    return path


@pytest.fixture()
def tmp_coco_json(tmp_path: Path) -> Path:
    """Create a minimal COCO-format JSON annotation file."""
    import json

    data = {
        "images": [
            {"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "img2.jpg", "width": 640, "height": 480},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 200, 100]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [300, 150, 150, 80]},
            {"id": 3, "image_id": 2, "category_id": 1, "bbox": [50, 50, 180, 120]},
        ],
        "categories": [{"id": 1, "name": "license_plate"}],
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture()
def sample_bbox():
    """Return a BoundingBox for quick testing."""
    from alpr_dataset.annotations.schema import BoundingBox

    return BoundingBox(x_min=10, y_min=20, x_max=50, y_max=80, class_id=0, class_name="plate")
