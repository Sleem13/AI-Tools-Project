"""Unit tests for dataset harmonization."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from alpr_dataset.annotations.schema import BoundingBox, ImageAnnotation
from alpr_dataset.harmonization.harmonizer import (
    harmonize_dataset,
    write_harmonization_metadata,
)


def _make_annotation(
    image_path: Path,
    boxes: list[BoundingBox],
    width: int = 128,
    height: int = 64,
    dataset_name: str = "test",
) -> ImageAnnotation:
    return ImageAnnotation(
        image_path=image_path,
        annotation_path=image_path.with_suffix(".txt"),
        image_width=width,
        image_height=height,
        dataset_name=dataset_name,
        boxes=boxes,
        source_format="yolo",
    )


@pytest.fixture()
def annotated_image(tmp_path: Path) -> Path:
    img = np.zeros((64, 128, 3), dtype=np.uint8)
    img[10:50, 20:100] = (100, 150, 200)
    path = tmp_path / "plate_001.jpg"
    cv2.imwrite(str(path), img)
    return path


class TestHarmonizeDataset:
    def test_creates_unified_files(
        self, annotated_image: Path, tmp_path: Path
    ) -> None:
        box = BoundingBox(x_min=20, y_min=10, x_max=100, y_max=50, class_id=0, class_name="plate")
        ann = _make_annotation(annotated_image, [box])
        output_root = tmp_path / "unified"

        records = harmonize_dataset(
            dataset_name="ds_a",
            annotations=[ann],
            unified_class_map={0: "plate"},
            local_class_map={0: "plate"},
            output_root=output_root,
        )

        assert len(records) == 1
        assert records[0].n_boxes == 1

        # Check output files exist
        img_out = output_root / "images" / "ds_a__plate_001.jpg"
        lbl_out = output_root / "labels" / "ds_a__plate_001.txt"
        assert img_out.exists()
        assert lbl_out.exists()

        # Check label content is valid YOLO
        content = lbl_out.read_text(encoding="utf-8").strip()
        parts = content.split()
        assert len(parts) == 5
        assert parts[0] == "0"  # class_id

    def test_class_id_remap(
        self, annotated_image: Path, tmp_path: Path
    ) -> None:
        box = BoundingBox(x_min=20, y_min=10, x_max=100, y_max=50, class_id=1, class_name="plate")
        ann = _make_annotation(annotated_image, [box])
        output_root = tmp_path / "unified"

        records = harmonize_dataset(
            dataset_name="ds_b",
            annotations=[ann],
            unified_class_map={0: "plate"},
            local_class_map={1: "plate"},
            output_root=output_root,
        )

        assert records[0].class_id_remap == {1: 0}
        lbl = (output_root / "labels" / "ds_b__plate_001.txt").read_text(encoding="utf-8")
        assert lbl.strip().startswith("0 ")

    def test_unmapped_classes_are_dropped(
        self, annotated_image: Path, tmp_path: Path
    ) -> None:
        box = BoundingBox(x_min=20, y_min=10, x_max=100, y_max=50, class_id=5, class_name="car")
        ann = _make_annotation(annotated_image, [box])
        output_root = tmp_path / "unified"

        records = harmonize_dataset(
            dataset_name="ds_c",
            annotations=[ann],
            unified_class_map={0: "plate"},
            local_class_map={5: "car"},
            output_root=output_root,
        )

        assert records[0].n_boxes == 0
        lbl = (output_root / "labels" / "ds_c__plate_001.txt").read_text(encoding="utf-8")
        assert lbl.strip() == ""

    def test_multiple_boxes(
        self, annotated_image: Path, tmp_path: Path
    ) -> None:
        boxes = [
            BoundingBox(x_min=10, y_min=5, x_max=50, y_max=30, class_id=0, class_name="plate"),
            BoundingBox(x_min=60, y_min=20, x_max=110, y_max=55, class_id=0, class_name="plate"),
        ]
        ann = _make_annotation(annotated_image, [boxes[0], boxes[1]])
        output_root = tmp_path / "unified"

        records = harmonize_dataset(
            dataset_name="ds_d",
            annotations=[ann],
            unified_class_map={0: "plate"},
            local_class_map={0: "plate"},
            output_root=output_root,
        )

        assert records[0].n_boxes == 2

    def test_skips_unreadable_image(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.jpg"
        ann = _make_annotation(missing, [
            BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10, class_id=0, class_name="plate")
        ])
        output_root = tmp_path / "unified"

        records = harmonize_dataset(
            dataset_name="ds_e",
            annotations=[ann],
            unified_class_map={0: "plate"},
            local_class_map={0: "plate"},
            output_root=output_root,
        )

        assert len(records) == 0

    def test_yolo_normalization_in_label(
        self, annotated_image: Path, tmp_path: Path
    ) -> None:
        box = BoundingBox(x_min=20, y_min=10, x_max=100, y_max=50, class_id=0, class_name="plate")
        ann = _make_annotation(annotated_image, [box], width=128, height=64)
        output_root = tmp_path / "unified"

        harmonize_dataset(
            dataset_name="ds_f",
            annotations=[ann],
            unified_class_map={0: "plate"},
            local_class_map={0: "plate"},
            output_root=output_root,
        )

        lbl = (output_root / "labels" / "ds_f__plate_001.txt").read_text(encoding="utf-8").strip()
        vals = lbl.split()
        cx, cy, w, h = float(vals[1]), float(vals[2]), float(vals[3]), float(vals[4])
        assert 0 <= cx <= 1
        assert 0 <= cy <= 1
        assert 0 < w <= 1
        assert 0 < h <= 1


class TestWriteHarmonizationMetadata:
    def test_writes_json(
        self, annotated_image: Path, tmp_path: Path
    ) -> None:
        box = BoundingBox(x_min=20, y_min=10, x_max=100, y_max=50, class_id=0, class_name="plate")
        ann = _make_annotation(annotated_image, [box])
        output_root = tmp_path / "unified"

        records = harmonize_dataset(
            dataset_name="ds_g",
            annotations=[ann],
            unified_class_map={0: "plate"},
            local_class_map={0: "plate"},
            output_root=output_root,
        )

        meta_dir = tmp_path / "metadata"
        meta_path = write_harmonization_metadata(records, meta_dir)
        assert meta_path.exists()

        import json
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        assert data["n_files_harmonized"] == 1
        assert len(data["records"]) == 1
