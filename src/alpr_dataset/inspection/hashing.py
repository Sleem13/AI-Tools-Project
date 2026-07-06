"""Exact and near-duplicate image detection via hashing.

Used by both Part 1 (dataset summary) and Part 3 (quality assessment).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import imagehash
from PIL import Image

from alpr_dataset.io_utils import md5_of_file

logger = logging.getLogger("alpr_dataset")


@dataclass
class DuplicateGroups:
    exact_duplicates: dict[str, list[Path]]  # md5 -> paths
    near_duplicates: list[tuple[Path, Path, int]]  # (path_a, path_b, hamming_distance)


def compute_exact_duplicates(paths: list[Path]) -> dict[str, list[Path]]:
    """Group files by MD5 hash; returns only groups with 2+ members (true duplicates)."""
    buckets: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        try:
            digest = md5_of_file(p)
            buckets[digest].append(p)
        except OSError as exc:
            logger.warning("Could not hash file %s: %s", p, exc)
    return {k: v for k, v in buckets.items() if len(v) > 1}


def compute_perceptual_hashes(paths: list[Path]) -> dict[Path, imagehash.ImageHash]:
    """Compute perceptual (phash) hashes for near-duplicate detection."""
    hashes: dict[Path, imagehash.ImageHash] = {}
    for p in paths:
        try:
            with Image.open(p) as im:
                hashes[p] = imagehash.phash(im)
        except Exception as exc:
            logger.warning("Could not compute phash for %s: %s", p, exc)
    return hashes


def find_near_duplicates(
    paths: list[Path], hamming_threshold: int = 5
) -> list[tuple[Path, Path, int]]:
    """Find near-duplicate image pairs using perceptual hash Hamming distance.

    O(n^2) comparison — fine for dataset-scale inspection reports; for very
    large datasets (100k+), consider bucketing hashes before comparing.
    """
    hashes = compute_perceptual_hashes(paths)
    items = list(hashes.items())
    near_dupes: list[tuple[Path, Path, int]] = []
    for i in range(len(items)):
        path_a, hash_a = items[i]
        for j in range(i + 1, len(items)):
            path_b, hash_b = items[j]
            distance = hash_a - hash_b
            if distance <= hamming_threshold:
                near_dupes.append((path_a, path_b, int(distance)))
    return near_dupes


def find_duplicates(
    paths: list[Path], hamming_threshold: int = 5
) -> DuplicateGroups:
    exact = compute_exact_duplicates(paths)
    exact_paths = {p for group in exact.values() for p in group}
    remaining = [p for p in paths if p not in exact_paths]
    near = find_near_duplicates(remaining, hamming_threshold=hamming_threshold)
    return DuplicateGroups(exact_duplicates=exact, near_duplicates=near)
