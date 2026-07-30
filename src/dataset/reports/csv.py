from __future__ import annotations

import csv
from pathlib import Path

from ..inspector.metadata import ImageMetadata
from ..inspector.scanner import ScanResult


class CSVReportGenerator:
    @staticmethod
    def generate_inventory(
        scan_results: list[ScanResult],
        metadata_list: list[list[ImageMetadata]],
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
                "mode",
                "md5",
                "phash",
                "path",
            ])

            for scan, metadatas in zip(scan_results, metadata_list, strict=False):
                for meta in metadatas:
                    writer.writerow([
                        scan.dataset_name,
                        meta.filename,
                        meta.stem,
                        meta.extension,
                        meta.size_bytes,
                        meta.width if meta.width is not None else "",
                        meta.height if meta.height is not None else "",
                        meta.mode if meta.mode is not None else "",
                        meta.md5 if meta.md5 is not None else "",
                        meta.phash if meta.phash is not None else "",
                        str(meta.path),
                    ])

        return output_path
