from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from .metadata import MetadataExtractor
from .scanner import DatasetScanner, ScanResult
from .tree import DirectoryTreeBuilder
from .validator import DatasetValidator, ValidationSummary


class DatasetInspector:
    def __init__(
        self,
        ignore_hidden: bool = True,
        ignore_cache: bool = True,
        use_tqdm: bool = True,
        near_duplicate_threshold: int = 8,
    ) -> None:
        self.scanner = DatasetScanner(
            ignore_hidden=ignore_hidden,
            ignore_cache=ignore_cache,
            use_tqdm=use_tqdm,
        )
        self.metadata_extractor = MetadataExtractor(
            compute_md5_flag=True,
            compute_phash_flag=True,
            use_tqdm=use_tqdm,
        )
        self.tree_builder = DirectoryTreeBuilder(
            ignore_hidden=ignore_hidden,
            ignore_cache=ignore_cache,
        )
        self.validator = DatasetValidator(
            use_tqdm=use_tqdm,
            near_duplicate_threshold=near_duplicate_threshold,
        )
        self.console = Console()

    def inspect(self, dataset_path: Path) -> dict:
        scan_result = self.scanner.scan(dataset_path)

        tree = self.tree_builder.build(dataset_path)
        tree_text = DirectoryTreeBuilder.render(tree)

        metadata = self.metadata_extractor.extract_many(scan_result.image_files)

        validation = self.validator.validate(scan_result)

        return {
            "scan": scan_result,
            "tree": tree_text,
            "metadata": metadata,
            "validation": validation,
        }

    def inspect_all(self, raw_path: Path) -> list[dict]:
        results: list[dict] = []
        scan_results = self.scanner.scan_all(raw_path)
        for scan_result in scan_results:
            tree = self.tree_builder.build(scan_result.root_path)
            tree_text = DirectoryTreeBuilder.render(tree)

            metadata = self.metadata_extractor.extract_many(scan_result.image_files)

            validation = self.validator.validate(scan_result)

            results.append({
                "scan": scan_result,
                "tree": tree_text,
                "metadata": metadata,
                "validation": validation,
            })
        return results

    def print_summary(self, results: list[dict]) -> None:
        table = Table(title="Dataset Inspection Summary")
        table.add_column("Dataset", style="cyan")
        table.add_column("Images", justify="right")
        table.add_column("Annotations", justify="right")
        table.add_column("Format", style="magenta")
        table.add_column("Issues", justify="right", style="red")

        for r in results:
            scan: ScanResult = r["scan"]
            validation: ValidationSummary = r["validation"]
            table.add_row(
                scan.dataset_name,
                str(scan.total_images),
                str(scan.total_annotations),
                scan.annotation_format.value,
                str(validation.total_issues),
            )

        self.console.print(table)
