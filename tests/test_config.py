"""Unit tests for configuration dataclasses and loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from alpr_dataset.config import (
    DatasetSpec,
    PreprocessingConfig,
    SplitConfig,
    load_yaml,
)


class TestDatasetSpec:
    def test_from_dict_minimal(self, tmp_path: Path) -> None:
        d = {"name": "ds_a", "root": "data/raw/ds_a"}
        spec = DatasetSpec.from_dict(d, project_root=tmp_path)
        assert spec.name == "ds_a"
        assert spec.root == tmp_path / "data/raw/ds_a"
        assert spec.annotation_format == "unknown"
        assert spec.class_map == {}

    def test_from_dict_full(self, tmp_path: Path) -> None:
        d = {
            "name": "ds_b",
            "root": "data/raw/ds_b",
            "annotation_format": "yolo",
            "image_subdir": "images",
            "annotation_subdir": "labels",
            "class_map": {"0": "plate"},
            "notes": "test dataset",
        }
        spec = DatasetSpec.from_dict(d, project_root=tmp_path)
        assert spec.annotation_format == "yolo"
        assert spec.image_subdir == "images"
        assert spec.class_map == {0: "plate"}
        assert spec.notes == "test dataset"


class TestPreprocessingConfig:
    def test_from_dict_default(self) -> None:
        cfg = PreprocessingConfig.from_dict({})
        assert cfg.target_size == (640, 640)
        assert cfg.steps == []
        assert cfg.save_before_after is True

    def test_from_dict_with_steps(self) -> None:
        d = {
            "target_size": [800, 600],
            "steps": [
                {"name": "clahe", "enabled": True, "params": {"clip_limit": 3.0}},
                {"name": "denoise", "enabled": False},
            ],
        }
        cfg = PreprocessingConfig.from_dict(d)
        assert cfg.target_size == (800, 600)
        assert len(cfg.steps) == 2
        assert cfg.steps[0].name == "clahe"
        assert cfg.steps[0].params == {"clip_limit": 3.0}
        assert cfg.steps[1].enabled is False


class TestSplitConfig:
    def test_valid_ratios(self) -> None:
        cfg = SplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        cfg.validate()  # should not raise

    def test_invalid_ratios_raise(self) -> None:
        cfg = SplitConfig(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)
        with pytest.raises(ValueError, match=r"must sum to 1\.0"):
            cfg.validate()

    def test_from_dict_valid(self) -> None:
        d = {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1, "seed": 7}
        cfg = SplitConfig.from_dict(d)
        assert cfg.train_ratio == 0.8
        assert cfg.seed == 7

    def test_from_dict_invalid_raises(self) -> None:
        d = {"train_ratio": 0.5, "val_ratio": 0.3, "test_ratio": 0.3}
        with pytest.raises(ValueError):
            SplitConfig.from_dict(d)

    def test_defaults(self) -> None:
        cfg = SplitConfig()
        assert cfg.train_ratio == 0.7
        assert cfg.val_ratio == 0.15
        assert cfg.test_ratio == 0.15
        assert cfg.seed == 42
        assert cfg.stratify_by_class is True


class TestLoadYaml:
    def test_load_valid_yaml(self, tmp_path: Path) -> None:
        p = tmp_path / "test.yaml"
        p.write_text("key: value\nlist:\n  - 1\n  - 2\n")
        data = load_yaml(p)
        assert data["key"] == "value"
        assert data["list"] == [1, 2]

    def test_load_empty_yaml(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.yaml"
        p.write_text("")
        data = load_yaml(p)
        assert data == {}
