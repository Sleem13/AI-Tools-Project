"""Regression tests for detection-training CLI defaults."""

from scripts.train_detection import DEFAULT_CONFIGS


def test_bare_plate_stage_defaults_to_master_plate_config() -> None:
    assert DEFAULT_CONFIGS["plate"].name == "master_plate_detection.yaml"
