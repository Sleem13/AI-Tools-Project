from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from src.dataset import DatasetInspector
from src.dataset.inspector.metadata import ImageMetadata
from src.dataset.inspector.scanner import ScanResult
from src.dataset.reports.csv import CSVReportGenerator
from src.dataset.reports.json import JSONReportGenerator
from src.dataset.reports.markdown import MarkdownReportGenerator
from src.dataset.reports.metadata import MetadataReportGenerator
from src.dataset.reports.statistics import StatisticsReportGenerator, compute_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset Discovery and Validation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--raw-path",
        type=str,
        default=None,
        help="Override raw data path from config",
    )
    parser.add_argument(
        "--reports-path",
        type=str,
        default=None,
        help="Override reports path from config",
    )
    parser.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable progress bars",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_cfg = config.get("dataset", {})
    quality_cfg = config.get("quality", {})

    raw_path = Path(args.raw_path) if args.raw_path else Path(dataset_cfg.get("raw_path", "data/raw"))
    reports_path = Path(args.reports_path) if args.reports_path else Path(dataset_cfg.get("reports_path", "reports"))
    use_tqdm = not args.no_tqdm

    inspector = DatasetInspector(
        use_tqdm=use_tqdm,
        near_duplicate_threshold=quality_cfg.get("near_duplicate_threshold", 8),
    )

    logger.info("Starting dataset inspection for: %s", raw_path)
    results = inspector.inspect_all(raw_path)

    inspector.print_summary(results)

    scan_results: list[ScanResult] = [r["scan"] for r in results]
    metadata_lists: list[list] = [r["metadata"] for r in results]
    tree_texts: list[str] = [r["tree"] for r in results]
    validation_summaries = [(r["scan"].dataset_name, r["validation"]) for r in results]

    reports_eda = reports_path / "eda"
    reports_eda.mkdir(parents=True, exist_ok=True)

    CSVReportGenerator.generate_inventory(scan_results, metadata_lists, reports_eda / "dataset_inventory.csv")
    logger.info("Generated: %s", reports_eda / "dataset_inventory.csv")

    JSONReportGenerator.generate_quality_report(validation_summaries, reports_eda / "quality_report.json")
    logger.info("Generated: %s", reports_eda / "quality_report.json")

    MarkdownReportGenerator.generate_inventory_report(
        scan_results, metadata_lists, tree_texts, reports_eda / "dataset_inventory.md",
    )
    logger.info("Generated: %s", reports_eda / "dataset_inventory.md")

    MarkdownReportGenerator.generate_quality_report(
        validation_summaries, tree_texts, reports_eda / "quality_report.md",
    )
    logger.info("Generated: %s", reports_eda / "quality_report.md")

    all_metadata: list[tuple[str, list[ImageMetadata]]] = [
        (r["scan"].dataset_name, r["metadata"]) for r in results
    ]

    MetadataReportGenerator.generate_metadata_csv(all_metadata, reports_eda / "metadata.csv")
    logger.info("Generated: %s", reports_eda / "metadata.csv")

    stats_list = [
        compute_stats(r["scan"].dataset_name, r["metadata"], r["scan"])
        for r in results
    ]

    StatisticsReportGenerator.generate_csv(stats_list, reports_eda / "dataset_statistics.csv")
    logger.info("Generated: %s", reports_eda / "dataset_statistics.csv")

    StatisticsReportGenerator.generate_markdown(stats_list, reports_eda / "dataset_statistics.md")
    logger.info("Generated: %s", reports_eda / "dataset_statistics.md")

    total_images = sum(s.total_images for s in scan_results)
    total_annotations = sum(s.total_annotations for s in scan_results)
    total_issues = sum(s.total_issues for _, s in validation_summaries)

    logger.info(
        "Inspection complete. %d images, %d annotations, %d issues found across %d dataset(s).",
        total_images,
        total_annotations,
        total_issues,
        len(results),
    )


if __name__ == "__main__":
    main()
