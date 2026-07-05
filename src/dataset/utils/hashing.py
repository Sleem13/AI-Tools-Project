from __future__ import annotations

import hashlib
from pathlib import Path

from imagehash import phash
from PIL import Image as PILImage


def compute_md5(file_path: Path, chunk_size: int = 65536) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def compute_phash(file_path: Path, hash_size: int = 8) -> str:
    with PILImage.open(file_path) as img:
        gray = img.convert("L")
        return str(phash(gray, hash_size=hash_size))


def find_exact_duplicates(
    files: list[Path],
    chunk_size: int = 65536,
) -> dict[str, list[Path]]:
    hash_map: dict[str, list[Path]] = {}
    for file_path in files:
        md5 = compute_md5(file_path, chunk_size=chunk_size)
        hash_map.setdefault(md5, []).append(file_path)
    return {h: paths for h, paths in hash_map.items() if len(paths) > 1}


def find_near_duplicates(
    files: list[Path],
    threshold: int = 8,
    hash_size: int = 8,
) -> list[tuple[Path, Path, int]]:
    pairs: list[tuple[Path, Path, int]] = []
    hashes: list[tuple[Path, int]] = []

    for file_path in files:
        try:
            h = int(compute_phash(file_path, hash_size=hash_size), 16)
            hashes.append((file_path, h))
        except Exception:
            continue

    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            diff = bin(hashes[i][1] ^ hashes[j][1]).count("1")
            if diff <= threshold:
                pairs.append((hashes[i][0], hashes[j][0], diff))

    return pairs
