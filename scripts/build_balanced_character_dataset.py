#!/usr/bin/env python3
"""Build a deterministic, training-only manifest for rare character classes."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "raw" / "dataset_Charcters_ready_plates" / "data.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "character_balanced"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def build_balanced_character_dataset(
    source_yaml: Path = DEFAULT_SOURCE,
    output_dir: Path = DEFAULT_OUTPUT,
    target_instances: int | None = None,
    max_extra_per_image: int = 2,
) -> dict[str, Any]:
    """Write an oversampled train manifest while leaving val/test untouched.

    Images are added greedily according to their contribution to classes that are
    still below the target. Ties are resolved by path, making the output stable.
    """
    source_yaml = source_yaml.resolve()
    config = yaml.safe_load(source_yaml.read_text(encoding="utf-8")) or {}
    names = _normalise_names(config.get("names", {}))
    split_paths = {
        split: _resolve_split(source_yaml, config, split)
        for split in ("train", "val", "test")
        if config.get(split)
    }
    train_images = _list_images(split_paths["train"])
    samples = [_read_sample(image) for image in train_images]
    before = Counter()
    for _, counts in samples:
        before.update(counts)

    present_counts = [before[class_id] for class_id in names if before[class_id] > 0]
    if not present_counts:
        raise RuntimeError(f"No YOLO labels were found for {source_yaml}")
    target = int(target_instances or statistics.median(present_counts))
    effective = before.copy()
    extras = Counter()
    repeated: list[Path] = []

    while True:
        best: tuple[float, str, Path, Counter[int]] | None = None
        for image, counts in samples:
            if extras[image] >= max_extra_per_image:
                continue
            score = sum(
                min(count, target - effective[class_id]) / max(before[class_id], 1)
                for class_id, count in counts.items()
                if effective[class_id] < target
            )
            if score <= 0:
                continue
            candidate = (score, image.as_posix(), image, counts)
            if best is None or candidate[0] > best[0] or (
                candidate[0] == best[0] and candidate[1] < best[1]
            ):
                best = candidate
        if best is None:
            break
        _, _, image, counts = best
        repeated.append(image)
        extras[image] += 1
        effective.update(counts)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "train.txt"
    manifest_entries = [*train_images, *repeated]
    manifest_path.write_text(
        "\n".join(path.resolve().as_posix() for path in manifest_entries) + "\n",
        encoding="utf-8",
    )

    balanced_yaml = output_dir / "data.yaml"
    balanced_config: dict[str, Any] = {
        "path": source_yaml.parent.resolve().as_posix(),
        "train": manifest_path.resolve().as_posix(),
        "val": split_paths["val"].resolve().as_posix(),
        "nc": len(names),
        "names": [names[index] for index in range(len(names))],
    }
    if "test" in split_paths:
        balanced_config["test"] = split_paths["test"].resolve().as_posix()
    balanced_yaml.write_text(
        yaml.safe_dump(balanced_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    report = {
        "source_yaml": source_yaml.as_posix(),
        "balanced_yaml": balanced_yaml.resolve().as_posix(),
        "target_instances": target,
        "max_extra_per_image": max_extra_per_image,
        "original_train_images": len(train_images),
        "balanced_train_entries": len(manifest_entries),
        "repeated_entries": len(repeated),
        "classes": {
            names[class_id]: {
                "id": class_id,
                "before": before[class_id],
                "effective_after": effective[class_id],
            }
            for class_id in names
        },
    }
    (output_dir / "balance_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _normalise_names(names: list[str] | dict[int | str, str]) -> dict[int, str]:
    if isinstance(names, list):
        return dict(enumerate(names))
    return {int(class_id): name for class_id, name in names.items()}


def _resolve_split(source_yaml: Path, config: dict[str, Any], split: str) -> Path:
    value = Path(str(config[split]))
    if value.is_absolute():
        resolved = value
    else:
        configured_root = Path(str(config.get("path", ".")))
        root = configured_root if configured_root.is_absolute() else source_yaml.parent / configured_root
        resolved = (root / value).resolve()
        if not resolved.exists():
            trimmed = Path(*[part for part in value.parts if part not in {"..", "."}])
            resolved = (root / trimmed).resolve()
    if not resolved.exists():
        raise RuntimeError(f"Missing {split} split: {resolved}")
    return resolved


def _list_images(split_path: Path) -> list[Path]:
    if split_path.is_file():
        entries = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        images = [Path(item) if Path(item).is_absolute() else split_path.parent / item for item in entries]
    else:
        images = [path for path in split_path.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES]
    return sorted((path.resolve() for path in images), key=lambda path: path.as_posix())


def _read_sample(image: Path) -> tuple[Path, Counter[int]]:
    parts = list(image.parts)
    try:
        image_dir_index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError as exc:
        raise RuntimeError(f"Image is not inside an images directory: {image}") from exc
    parts[image_dir_index] = "labels"
    label = Path(*parts).with_suffix(".txt")
    counts: Counter[int] = Counter()
    if label.is_file():
        for line in label.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if fields:
                counts[int(fields[0])] += 1
    return image, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-instances", type=int, default=None)
    parser.add_argument(
        "--max-extra-per-image",
        type=int,
        default=2,
        help="Maximum extra appearances for one source image (default: 2)",
    )
    args = parser.parse_args()
    report = build_balanced_character_dataset(
        args.source, args.output, args.target_instances, args.max_extra_per_image
    )
    print(f"Balanced dataset: {report['balanced_yaml']}")
    print(
        f"Train entries: {report['original_train_images']} -> "
        f"{report['balanced_train_entries']} (target={report['target_instances']} instances/class)"
    )


if __name__ == "__main__":
    main()
