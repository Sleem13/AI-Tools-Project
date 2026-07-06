"""Unit tests for the reproducible train/val/test splitter."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpr_dataset.annotations.schema import BoundingBox, ImageAnnotation
from alpr_dataset.config import SplitConfig
from alpr_dataset.splitting.splitter import stratified_split


def _make_annotations(n: int, class_name: str) -> list[ImageAnnotation]:
    return [
        ImageAnnotation(
            image_path=Path(f"{class_name}_{i}.jpg"),
            annotation_path=Path(f"{class_name}_{i}.txt"),
            image_width=100,
            image_height=100,
            dataset_name="test",
            boxes=[BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10, class_id=0, class_name=class_name)],
        )
        for i in range(n)
    ]


def test_split_ratios_approximately_correct():
    annotations = _make_annotations(100, "plate")
    config = SplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
    result = stratified_split(annotations, config)
    summary = result.summary()
    assert summary["n_total"] == 100
    assert 65 <= summary["n_train"] <= 75
    assert 10 <= summary["n_val"] <= 20
    assert 10 <= summary["n_test"] <= 20


def test_split_is_reproducible_with_same_seed():
    annotations = _make_annotations(50, "plate")
    config = SplitConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=7)
    result_a = stratified_split(annotations, config)
    result_b = stratified_split(annotations, config)
    assert [a.image_path for a in result_a.train] == [b.image_path for b in result_b.train]


def test_split_preserves_class_balance():
    annotations = _make_annotations(60, "class_a") + _make_annotations(40, "class_b")
    config = SplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=1, stratify_by_class=True)
    result = stratified_split(annotations, config)

    def count_class(items, name):
        return sum(1 for a in items if a.boxes[0].class_name == name)

    # Both classes should appear in train (stratification keeps minority represented).
    assert count_class(result.train, "class_a") > 0
    assert count_class(result.train, "class_b") > 0


def test_invalid_ratios_raise():
    config = SplitConfig(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)
    try:
        config.validate()
        assert False, "Expected ValueError for ratios not summing to 1.0"
    except ValueError:
        pass
