"""Configuration schema and loaders for the ALPR dataset pipeline.

All paths are resolved via pathlib and are never hardcoded — every path
originates from a YAML config file or a CLI argument.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# --------------------------------------------------------------------------- #
# Dataset-level configuration
# --------------------------------------------------------------------------- #
@dataclass
class DatasetSpec:
    """Describes a single raw source dataset."""

    name: str
    root: Path
    annotation_format: str  # one of: "yolo", "voc_xml", "coco_json", "csv", "unknown"
    image_subdir: str | None = None
    annotation_subdir: str | None = None
    class_map: dict[int, str] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any], project_root: Path) -> DatasetSpec:
        return cls(
            name=d["name"],
            root=project_root / d["root"],
            annotation_format=d.get("annotation_format", "unknown"),
            image_subdir=d.get("image_subdir"),
            annotation_subdir=d.get("annotation_subdir"),
            class_map={int(k): v for k, v in d.get("class_map", {}).items()},
            notes=d.get("notes", ""),
        )


# --------------------------------------------------------------------------- #
# Preprocessing configuration
# --------------------------------------------------------------------------- #
@dataclass
class PreprocessingStepConfig:
    """A single named preprocessing step with arbitrary keyword parameters."""

    name: str
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessingConfig:
    target_size: tuple[int, int] = (640, 640)
    steps: list[PreprocessingStepConfig] = field(default_factory=list)
    save_before_after: bool = True
    before_after_sample_count: int = 20

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PreprocessingConfig:
        steps = [
            PreprocessingStepConfig(
                name=s["name"],
                enabled=s.get("enabled", True),
                params=s.get("params", {}),
            )
            for s in d.get("steps", [])
        ]
        size = d.get("target_size", [640, 640])
        return cls(
            target_size=(size[0], size[1]),
            steps=steps,
            save_before_after=d.get("save_before_after", True),
            before_after_sample_count=d.get("before_after_sample_count", 20),
        )


# --------------------------------------------------------------------------- #
# Split configuration
# --------------------------------------------------------------------------- #
@dataclass
class SplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    stratify_by_class: bool = True
    copy_files: bool = False

    def validate(self) -> None:
        total = round(self.train_ratio + self.val_ratio + self.test_ratio, 6)
        if total != 1.0:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total} "
                f"(train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio})"
            )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SplitConfig:
        cfg = cls(
            train_ratio=d.get("train_ratio", 0.7),
            val_ratio=d.get("val_ratio", 0.15),
            test_ratio=d.get("test_ratio", 0.15),
            seed=d.get("seed", 42),
            stratify_by_class=d.get("stratify_by_class", True),
            copy_files=d.get("copy_files", False),
        )
        cfg.validate()
        return cfg


# --------------------------------------------------------------------------- #
# Top-level pipeline configuration
# --------------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    project_root: Path
    data_raw_dir: Path
    data_processed_dir: Path
    reports_dir: Path
    logs_dir: Path
    datasets: list[DatasetSpec]
    duplicate_hash_threshold: int = 5  # perceptual hash hamming distance
    blur_threshold: float = 100.0  # variance-of-laplacian below = blurry
    random_seed: int = 42
    unified_class_map: dict[int, str] = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: Path, datasets_path: Path) -> PipelineConfig:
        config_path = Path(config_path)
        datasets_path = Path(datasets_path)
        project_root = config_path.resolve().parent.parent

        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        with open(datasets_path, encoding="utf-8") as f:
            raw_datasets = yaml.safe_load(f)

        datasets = [
            DatasetSpec.from_dict(d, project_root) for d in raw_datasets["datasets"]
        ]

        paths = raw.get("paths", {})
        return cls(
            project_root=project_root,
            data_raw_dir=project_root / paths.get("data_raw", "data/raw"),
            data_processed_dir=project_root / paths.get("data_processed", "data/processed"),
            reports_dir=project_root / paths.get("reports", "reports"),
            logs_dir=project_root / paths.get("logs", "reports/logs"),
            datasets=datasets,
            duplicate_hash_threshold=raw.get("duplicate_hash_threshold", 5),
            blur_threshold=raw.get("blur_threshold", 100.0),
            random_seed=raw.get("random_seed", 42),
            unified_class_map={
                int(k): v for k, v in raw_datasets.get("unified_class_map", {}).items()
            },
        )

    def preprocessing_config(self, preprocessing_path: Path) -> PreprocessingConfig:
        with open(preprocessing_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return PreprocessingConfig.from_dict(raw)

    def split_config(self, preprocessing_path: Path) -> SplitConfig:
        with open(preprocessing_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return SplitConfig.from_dict(raw.get("split", {}))


def load_yaml(path: Path) -> dict[str, Any]:
    """Generic YAML loader used across scripts."""
    with open(Path(path), encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
