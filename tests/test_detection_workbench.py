"""Tests for notebook-derived Master Plate workbench logic."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from src.detection.workbench import (
    discover_latest_run,
    inspect_dataset,
    natural_high_resolution_plate,
    read_training_history,
    resolve_dataset_image,
    sample_images,
)


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((8, 12, 3), dtype=np.uint8))


def test_inspects_master_dataset_splits(tmp_path: Path) -> None:
    root = tmp_path / "Master_Plate_Dataset"
    _write_image(root / "images" / "train" / "one.jpg")
    _write_image(root / "images" / "val" / "two.png")
    (root / "labels" / "train").mkdir(parents=True)
    (root / "labels" / "train" / "one.txt").write_text("0 0.5 0.5 0.2 0.2", encoding="utf-8")
    (root / "data.yaml").write_text("names: [license_plate]", encoding="utf-8")

    summary = inspect_dataset(root)

    assert summary["splits"]["train"] == {"images": 1, "labels": 1}
    assert summary["splits"]["val"] == {"images": 1, "labels": 0}
    assert summary["total_images"] == 2
    assert summary["ready"] is True


def test_samples_are_seeded_and_bounded(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    for index in range(3):
        _write_image(root / "images" / "train" / f"{index}.jpg")

    first = sample_images(root, "train", count=5, seed=7)
    second = sample_images(root, "train", count=5, seed=7)

    assert first == second
    assert len(first) == 3


def test_dataset_image_resolution_blocks_traversal(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_image(root / "images" / "train" / "nested" / "plate.jpg")

    assert resolve_dataset_image(root, "train", "nested/plate.jpg").name == "plate.jpg"
    with pytest.raises(ValueError, match="escapes"):
        resolve_dataset_image(root, "train", "../../secret.jpg")


def test_natural_plate_enhancement_uses_requested_scale() -> None:
    crop = np.zeros((10, 20, 3), dtype=np.uint8)

    enhanced = natural_high_resolution_plate(crop, scale_factor=4)

    assert enhanced.shape == (40, 80, 3)


def test_reads_ultralytics_training_history(tmp_path: Path) -> None:
    results = tmp_path / "results.csv"
    results.write_text(
        "epoch,train/box_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "0,0.8,0.91,0.92,0.95,0.75\n"
        "1,0.6,0.96,0.97,0.99,0.84\n",
        encoding="utf-8",
    )

    parsed = read_training_history(results)

    assert parsed["latest"]["epoch"] == 2
    assert parsed["latest"]["map50_95"] == pytest.approx(0.84)
    assert parsed["history"][0]["train_box_loss"] == pytest.approx(0.8)


def test_discovers_latest_training_run(tmp_path: Path) -> None:
    older = tmp_path / "older"
    newer = tmp_path / "newer"
    older.mkdir()
    newer.mkdir()
    (older / "results.csv").write_text("epoch\n0\n", encoding="utf-8")
    (newer / "results.csv").write_text("epoch\n0\n", encoding="utf-8")
    (newer / "results.csv").touch()

    assert discover_latest_run(tmp_path) == newer
