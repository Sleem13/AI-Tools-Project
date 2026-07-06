"""Part 7 — Train / validation / test splitting.

Supports arbitrary ratios (70/15/15, 80/10/10, custom), a fixed random
seed for reproducibility, and best-effort class-balance preservation via
stratification on each image's dominant (most frequent) box class.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from alpr_dataset.annotations.schema import ImageAnnotation
from alpr_dataset.config import SplitConfig
from alpr_dataset.io_utils import ensure_dir

logger = logging.getLogger("alpr_dataset")


@dataclass
class SplitResult:
    train: list[ImageAnnotation]
    val: list[ImageAnnotation]
    test: list[ImageAnnotation]

    def summary(self) -> dict:
        return {
            "n_train": len(self.train),
            "n_val": len(self.val),
            "n_test": len(self.test),
            "n_total": len(self.train) + len(self.val) + len(self.test),
        }


def _dominant_class(ann: ImageAnnotation) -> str:
    if not ann.boxes:
        return "__background__"
    counts = Counter(b.class_name or str(b.class_id) for b in ann.boxes)
    return counts.most_common(1)[0][0]


def stratified_split(
    annotations: list[ImageAnnotation], config: SplitConfig
) -> SplitResult:
    """Split annotations into train/val/test, stratifying by dominant class when possible."""
    config.validate()
    rng = random.Random(config.seed)

    if not config.stratify_by_class:
        shuffled = annotations[:]
        rng.shuffle(shuffled)
        return _slice_three_way(shuffled, config)

    groups: dict[str, list[ImageAnnotation]] = defaultdict(list)
    for ann in annotations:
        groups[_dominant_class(ann)].append(ann)

    train, val, test = [], [], []
    for class_name, items in groups.items():
        rng.shuffle(items)
        n = len(items)
        n_train = round(n * config.train_ratio)
        n_val = round(n * config.val_ratio)
        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])
        logger.debug(
            "Class '%s': %d items -> train=%d val=%d test=%d",
            class_name,
            n,
            n_train,
            n_val,
            n - n_train - n_val,
        )

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return SplitResult(train=train, val=val, test=test)


def _slice_three_way(items: list[ImageAnnotation], config: SplitConfig) -> SplitResult:
    n = len(items)
    n_train = round(n * config.train_ratio)
    n_val = round(n * config.val_ratio)
    return SplitResult(
        train=items[:n_train],
        val=items[n_train : n_train + n_val],
        test=items[n_train + n_val :],
    )


def write_split_manifests(result: SplitResult, output_dir: Path) -> dict[str, Path]:
    """Write train.txt / val.txt / test.txt (image paths) + a JSON manifest with full detail."""
    output_dir = ensure_dir(output_dir)
    paths: dict[str, Path] = {}

    for split_name, items in (("train", result.train), ("val", result.val), ("test", result.test)):
        txt_path = output_dir / f"{split_name}.txt"
        txt_path.write_text(
            "\n".join(str(a.image_path) for a in items), encoding="utf-8"
        )
        paths[split_name] = txt_path

    manifest = {
        "summary": result.summary(),
        "train": [str(a.image_path) for a in result.train],
        "val": [str(a.image_path) for a in result.val],
        "test": [str(a.image_path) for a in result.test],
    }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote split manifests -> %s (%s)", output_dir, result.summary())
    return paths


def materialize_split(
    result: SplitResult,
    label_lookup: dict[Path, Path],
    output_root: Path,
) -> None:
    """Physically copy images + labels into split_output/{train,val,test}/{images,labels}/.

    Only called when SplitConfig.copy_files is True — otherwise the split
    manifests (image path lists) are sufficient for most training loaders.
    """
    for split_name, items in (("train", result.train), ("val", result.val), ("test", result.test)):
        img_dir = ensure_dir(output_root / split_name / "images")
        lbl_dir = ensure_dir(output_root / split_name / "labels")
        for ann in items:
            src_img = Path(ann.image_path)
            if not src_img.exists():
                continue
            shutil.copy2(src_img, img_dir / src_img.name)
            label_path = label_lookup.get(src_img)
            if label_path and label_path.exists():
                shutil.copy2(label_path, lbl_dir / label_path.name)
    logger.info("Materialized split files under %s", output_root)
