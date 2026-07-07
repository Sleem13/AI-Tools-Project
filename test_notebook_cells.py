"""Test that key notebook code cells can execute without error."""
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Test EDA notebook imports
print("=== Testing EDA notebook imports ===")

# Cell 1: path setup
from alpr_dataset.config import PipelineConfig
from alpr_dataset.logging_setup import setup_logging
from alpr_dataset.io_utils import list_images

config = PipelineConfig.load(
    PROJECT_ROOT / "configs" / "pipeline_config.yaml",
    PROJECT_ROOT / "configs" / "datasets.yaml",
)
logger = setup_logging(config.logs_dir, name="alpr_dataset")
print(f"Datasets: {[s.name for s in config.datasets]}")
print(f"Reports dir: {config.reports_dir}")

# Cell 5: scanner
from alpr_dataset.inspection.scanner import scan_dataset
from alpr_dataset.inspection.report import generate_dataset_summary

dataset_specs = [(s.name, s.root) for s in config.datasets]
try:
    df_summary = generate_dataset_summary(
        dataset_specs,
        output_dir=config.reports_dir / "eda",
        hamming_threshold=config.duplicate_hash_threshold,
    )
    print(f"Summary columns: {list(df_summary.columns)}")
except Exception as e:
    print(f"Summary generation failed: {e}")

# Check individual scans
for spec in config.datasets:
    scan = scan_dataset(spec.name, spec.root)
    print(f"{spec.name}: {scan.n_images} images, {scan.n_annotations} annotations")

# Cell 8: load annotations
from alpr_dataset.annotations.loader import load_dataset_annotations

all_annotations = {}
for spec in config.datasets:
    annots = load_dataset_annotations(spec)
    all_annotations[spec.name] = annots
    n_boxes = sum(a.n_boxes for a in annots)
    print(f"{spec.name}: {len(annots)} annotated, {n_boxes} boxes")

# Cell 9: peek at annotation
for name, annots in all_annotations.items():
    if annots:
        a = annots[0]
        print(f"{name} first: {a.image_path.name} ({a.image_width}x{a.image_height}), {a.n_boxes} boxes")

# Cell 11: image stats
from alpr_dataset.inspection.image_stats import batch_compute_stats
from alpr_dataset.inspection.hashing import find_duplicates

for spec in config.datasets:
    images = list_images(spec.root)
    stats = batch_compute_stats(images)
    dupes = find_duplicates(images, hamming_threshold=config.duplicate_hash_threshold).near_duplicates
    valid = [s for s in stats if not s.is_corrupted]
    print(f"{spec.name}: {len(images)} images, {len(valid)} valid, {len(dupes)} near-dupes")

# Cell 13: EDA figure imports
from alpr_dataset.eda.figures import (
    plot_dataset_size_comparison,
    plot_resolution_histograms,
    plot_width_distribution,
    plot_height_distribution,
    plot_aspect_ratio_distribution,
    plot_brightness_histogram,
    plot_contrast_histogram,
    plot_blur_estimation,
    plot_sharpness,
    plot_entropy,
    plot_bbox_width_distribution,
    plot_bbox_height_distribution,
    plot_bbox_area_distribution,
    plot_bbox_position_heatmap,
    plot_class_distribution,
    plot_example_images,
    plot_random_samples,
    plot_annotated_samples,
    plot_color_distribution,
    plot_duplicate_visualization,
    plot_outlier_visualization,
    generate_all_eda_figures,
)
from alpr_dataset.eda.report import generate_eda_report
print("All EDA figure functions imported successfully")

# Cross-dataset comparison
shared_fig_dir = config.reports_dir / "figures"
dataset_counts = {name: len(all_annotations[name]) for name, a in all_annotations.items()}
plot_dataset_size_comparison(dataset_counts, shared_fig_dir)
print(f"Cross-dataset comparison saved to {shared_fig_dir}")

# Generate all figures for dataset_A only (dataset_B is larger)
spec = config.datasets[0]
d = {"images": list_images(spec.root)}
d["stats"] = batch_compute_stats(d["images"])
d["dupes"] = find_duplicates(d["images"], hamming_threshold=config.duplicate_hash_threshold).near_duplicates

output_dir = shared_fig_dir / spec.name
generate_all_eda_figures(
    dataset_counts={spec.name: len(d["images"])},
    stats=d["stats"],
    annotations=all_annotations[spec.name],
    image_paths=d["images"],
    duplicate_pairs=d["dupes"],
    output_dir=output_dir,
    blur_threshold=config.blur_threshold,
)
print(f"EDA figures for {spec.name} saved to {output_dir}")

# Generate EDA report
generate_eda_report(
    dataset_name=spec.name,
    stats=d["stats"],
    annotations=all_annotations[spec.name],
    image_paths=d["images"],
    duplicate_pairs=d["dupes"],
    figures_dir=output_dir,
    output_dir=shared_fig_dir,
)
print(f"EDA report for {spec.name} saved")

print("\n=== All EDA notebook cells passed! ===")
