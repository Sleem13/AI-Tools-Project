"""Unit tests for memory-safe Ultralytics training arguments."""

from src.detection.trainer import _build_training_args


def test_character_batch_pixel_budget_caps_high_resolution_batch() -> None:
    config = {
        "hyperparameters": {"batch": 32, "imgsz": 960},
        "training": {"max_batch_pixels": 8 * 960 * 960},
    }

    assert _build_training_args(config)["batch"] == 8


def test_batch_is_unchanged_without_an_opt_in_pixel_budget() -> None:
    config = {"hyperparameters": {"batch": 32, "imgsz": 960}}

    assert _build_training_args(config)["batch"] == 32
