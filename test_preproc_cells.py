"""Test preprocessing notebook code cells."""
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

print("=== Testing Preprocessing notebook imports ===")

from alpr_dataset.config import PipelineConfig
from alpr_dataset.logging_setup import setup_logging

config = PipelineConfig.load(
    PROJECT_ROOT / "configs" / "pipeline_config.yaml",
    PROJECT_ROOT / "configs" / "datasets.yaml",
)
prep_config = config.preprocessing_config(
    PROJECT_ROOT / "configs" / "preprocessing_config.yaml"
)

print(f"Datasets: {[s.name for s in config.datasets]}")
print(f"Target size: {prep_config.target_size}")
print(f"Steps enabled: {[s.name for s in prep_config.steps if s.enabled]}")

# Load annotations
from alpr_dataset.annotations.loader import load_dataset_annotations
all_annotations = {}
for spec in config.datasets:
    all_annotations[spec.name] = load_dataset_annotations(spec)
    print(f"{spec.name}: {len(all_annotations[spec.name])} annotations")

# Preprocessing pipeline
from alpr_dataset.preprocessing.pipeline import PreprocessingPipeline
pipeline = PreprocessingPipeline(prep_config)
for i, step in enumerate(prep_config.steps):
    if not step.enabled:
        continue
    print(f"  {i+1}. {step.name} enabled={step.enabled}")

# Test applying pipeline to one image
from alpr_dataset.io_utils import list_images, safe_read_image
from alpr_dataset.utils.viz_utils import bgr_to_rgb
import cv2
import numpy as np

sample_img_path = None
for spec in config.datasets:
    images = list_images(spec.root)
    if images:
        sample_img_path = images[0]
        break

if sample_img_path:
    original = safe_read_image(sample_img_path)
    processed = pipeline.apply(original)
    print(f"\nSample: {sample_img_path.name}")
    print(f"Original shape: {original.shape}, Processed shape: {processed.shape}")
    print(f"Original dtype: {original.dtype}, Processed dtype: {processed.dtype}")

print("\n=== All Preprocessing notebook cells passed! ===")
