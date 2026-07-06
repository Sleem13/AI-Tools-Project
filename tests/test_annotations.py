"""Unit tests for core annotation geometry and parsing logic.

Run with: pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from alpr_dataset.annotations.schema import BoundingBox, ImageAnnotation
from alpr_dataset.annotations.validators import validate_image_annotation


def test_bbox_geometry():
    box = BoundingBox(x_min=10, y_min=20, x_max=50, y_max=80, class_id=0)
    assert box.width == 40
    assert box.height == 60
    assert box.area == 2400
    assert box.center == (30, 50)


def test_bbox_normalization():
    box = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=50, class_id=0)
    cx, cy, w, h = box.to_normalized(200, 100)
    assert cx == pytest.approx(0.25)
    assert cy == pytest.approx(0.25)
    assert w == pytest.approx(0.5)
    assert h == pytest.approx(0.5)


def test_bbox_iou_identical():
    a = BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10, class_id=0)
    b = BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10, class_id=0)
    assert a.iou(b) == pytest.approx(1.0)


def test_bbox_iou_disjoint():
    a = BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10, class_id=0)
    b = BoundingBox(x_min=100, y_min=100, x_max=110, y_max=110, class_id=0)
    assert a.iou(b) == pytest.approx(0.0)


def test_validate_boxes_outside_image():
    ann = ImageAnnotation(
        image_path=Path("fake.jpg"),
        annotation_path=Path("fake.txt"),
        image_width=100,
        image_height=100,
        dataset_name="test",
        boxes=[BoundingBox(x_min=-5, y_min=0, x_max=50, y_max=50, class_id=0)],
    )
    issues = validate_image_annotation(ann)
    assert issues.boxes_outside_image == 1


def test_validate_zero_area_box():
    ann = ImageAnnotation(
        image_path=Path("fake.jpg"),
        annotation_path=Path("fake.txt"),
        image_width=100,
        image_height=100,
        dataset_name="test",
        boxes=[BoundingBox(x_min=10, y_min=10, x_max=10, y_max=50, class_id=0)],
    )
    issues = validate_image_annotation(ann)
    assert issues.zero_area_boxes == 1
    assert issues.boxes_outside_image == 1  # x_min >= x_max also trips this check


def test_validate_unknown_class_id():
    ann = ImageAnnotation(
        image_path=Path("fake.jpg"),
        annotation_path=Path("fake.txt"),
        image_width=100,
        image_height=100,
        dataset_name="test",
        boxes=[BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10, class_id=99)],
    )
    issues = validate_image_annotation(ann, valid_class_ids={0, 1})
    assert issues.unknown_class_ids == [99]


def test_validate_overlapping_boxes():
    ann = ImageAnnotation(
        image_path=Path("fake.jpg"),
        annotation_path=Path("fake.txt"),
        image_width=200,
        image_height=200,
        dataset_name="test",
        boxes=[
            BoundingBox(x_min=0, y_min=0, x_max=50, y_max=50, class_id=0),
            BoundingBox(x_min=1, y_min=1, x_max=51, y_max=51, class_id=0),
        ],
    )
    issues = validate_image_annotation(ann, overlap_iou_threshold=0.7)
    assert issues.overlapping_box_pairs == 1
