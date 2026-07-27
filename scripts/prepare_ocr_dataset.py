#!/usr/bin/env python3
"""Build OCR training data from the character detection dataset.

Reads plate crops and character-level YOLO annotations, derives plate text
in reading order (RTL), and writes ``data/ocr/labels.json`` + crop images.

Usage:
    python scripts/prepare_ocr_dataset.py
    python scripts/prepare_ocr_dataset.py --character-dataset data/raw/dataset_Charcters_ready_plates
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_NAMES = [
    "0", "1", "2", "3", "4", "5", "6", "7", "7aa", "8", "9",
    "Taa", "Thaa", "ain", "alif", "baa", "daad", "daal", "faa",
    "ghayn", "haa", "jeem", "kaaf", "khaa", "laam", "meem", "noon",
    "qaaf", "raa", "saad", "seen", "sheen", "taa", "thaa", "waw",
    "yaa", "zaal", "zay",
]

CHARACTER_GLYPHS = {
    "7aa": "ح", "Taa": "ط", "Thaa": "ظ", "ain": "ع", "alif": "ا",
    "baa": "ب", "daad": "ض", "daal": "د", "faa": "ف", "ghayn": "غ",
    "haa": "ه", "jeem": "ج", "kaaf": "ك", "khaa": "خ", "laam": "ل",
    "meem": "م", "noon": "ن", "qaaf": "ق", "raa": "ر", "saad": "ص",
    "seen": "س", "sheen": "ش", "taa": "ت", "thaa": "ث", "waw": "و",
    "yaa": "ي", "zaal": "ذ", "zay": "ز",
}

OCR_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ARABIC_TO_LATIN = {
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
}


def _class_to_char(class_id: int) -> str:
    if 0 <= class_id < len(CLASS_NAMES):
        name = CLASS_NAMES[class_id]
        glyph = CHARACTER_GLYPHS.get(name, name)
        if glyph in ARABIC_TO_LATIN:
            return ARABIC_TO_LATIN[glyph]
        if len(glyph) == 1 and glyph.upper() in OCR_CHARS:
            return glyph.upper()
        return ""
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OCR dataset from character detection annotations.")
    parser.add_argument(
        "--character-dataset", type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "dataset_Charcters_ready_plates",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "data" / "ocr",
    )
    args = parser.parse_args()

    crops_dir = args.output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    splits = [("train", "train"), ("val", "valid"), ("test", "test")]
    records: list[dict] = []

    for api_name, dir_name in splits:
        img_dir = args.character_dataset / dir_name / "images"
        lbl_dir = args.character_dataset / dir_name / "labels"
        if not img_dir.is_dir():
            print(f"Skipping {dir_name}/images — not found")
            continue

        for img_path in sorted(img_path for img_path in img_dir.iterdir() if img_path.suffix.lower() in IMAGE_SUFFIXES):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            characters: list[tuple[float, str]] = []
            if lbl_path.is_file():
                for line in lbl_path.read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    class_id = int(parts[0])
                    cx = float(parts[1])
                    ch = _class_to_char(class_id)
                    if ch:
                        characters.append((cx, ch))

            characters.sort(key=lambda item: item[0], reverse=True)
            plate_text = "".join(ch for _, ch in characters)

            crop_name = f"{api_name}_{img_path.name}"
            crop_path = crops_dir / crop_name
            shutil.copy2(str(img_path), str(crop_path))

            records.append({
                "image_path": crop_name,
                "source_image": str(img_path.relative_to(args.character_dataset)),
                "split": api_name,
                "text": plate_text,
                "confidence": 1.0,
            })

    labels_path = args.output_dir / "labels.json"
    labels_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Prepared {len(records)} OCR samples -> {args.output_dir}")
    with_text = sum(1 for r in records if r["text"])
    print(f"  {with_text} with text, {len(records) - with_text} empty")


if __name__ == "__main__":
    main()
