from __future__ import annotations

import csv
from pathlib import Path

from ..inspector.metadata import ImageMetadata


class MetadataReportGenerator:
    @staticmethod
    def generate_metadata_csv(
        all_metadata: list[tuple[str, list[ImageMetadata]]],
        output_path: Path,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "dataset",
                "filename",
                "stem",
                "extension",
                "size_bytes",
                "width",
                "height",
                "channels",
                "aspect_ratio",
                "resolution",
                "mode",
                "brightness",
                "contrast",
                "sharpness",
                "blur_score",
                "entropy",
                "md5",
                "phash",
                "path",
            ])

            for dataset_name, metadatas in all_metadata:
                for meta in metadatas:
                    writer.writerow([
                        dataset_name,
                        meta.filename,
                        meta.stem,
                        meta.extension,
                        meta.size_bytes,
                        _v(meta.width),
                        _v(meta.height),
                        _v(meta.channels),
                        _v(meta.aspect_ratio),
                        _v(meta.resolution),
                        meta.mode or "",
                        _v(meta.brightness),
                        _v(meta.contrast),
                        _v(meta.sharpness),
                        _v(meta.blur_score),
                        _v(meta.entropy),
                        meta.md5 or "",
                        meta.phash or "",
                        str(meta.path),
                    ])

        return output_path


def _v(val: object) -> str:
    if val is None:
        return ""
    return str(val)
