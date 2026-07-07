# Reusable Prompt: ALPR Dataset Pipeline — EDA & Preprocessing

## Task

Build a complete ALPR (Automatic License Plate Recognition) dataset preparation pipeline that:

1. **Inspects** raw datasets (scan folder structure, count files, detect formats, identify orphans)
2. **Exploratory Data Analysis** — generate 22 publication-quality figures (PNG+SVG) per dataset plus cross-dataset comparison
3. **Quality Assessment** — detect corrupt/blurry images, validate bounding boxes
4. **Harmonization** — convert all annotations to unified YOLO format
5. **Preprocessing** — apply configurable transforms (denoise → CLAHE → gamma → bilateral → sharpen → letterbox)
6. **Statistics** — compute pre/post stats with before/after comparison figures
7. **Stratified Split** — train/val/test split (70/15/15) stratified by class
8. **Orchestration** — single `run_full_pipeline.py` to execute all stages end-to-end
9. **Notebooks** — `eda.ipynb` and `preprocessing.ipynb` as supplementary walkthroughs

## Project Structure

```
project_root/
├── configs/
│   ├── pipeline_config.yaml      # Global paths, thresholds
│   ├── datasets.yaml             # Dataset specs (annotation format, subdirs, class maps)
│   └── preprocessing_config.yaml # Transform pipeline + split config
├── data/
│   └── processed/
│       ├── unified/              # Harmonized YOLO labels
│       ├── preprocessed/         # Transformed images + labels
│       └── split/                # Train/val/test splits
├── scripts/
│   ├── run_inspection.py
│   ├── run_eda.py
│   ├── run_quality_check.py
│   ├── run_harmonization.py
│   ├── run_preprocessing.py
│   ├── run_statistics.py
│   ├── run_split.py
│   └── run_full_pipeline.py
├── src/
│   └── alpr_dataset/
│       ├── config.py             # PipelineConfig, PreprocessingConfig dataclasses
│       ├── logging_setup.py
│       ├── io_utils.py           # safe_read_image (unicode path support)
│       ├── annotations/
│       │   ├── loader.py         # load_dataset_annotations() — integration point
│       │   └── schema.py         # ImageAnnotation, BoundingBox dataclasses
│       ├── inspection/
│       │   ├── scanner.py
│       │   ├── report.py
│       │   ├── image_stats.py    # ImageStats, compute_image_stats, batch_compute_stats
│       │   └── hashing.py        # find_duplicates
│       ├── eda/
│       │   ├── figures.py        # 22 plot functions, generate_all_eda_figures, save_figure
│       │   ├── quality.py        # compute_quality_report
│       │   └── report.py         # generate_eda_report
│       ├── harmonization/
│       │   └── harmonizer.py     # harmonize_dataset()
│       ├── preprocessing/
│       │   ├── pipeline.py       # PreprocessingPipeline, STEP_REGISTRY
│       │   ├── transforms.py     # 12 transform functions
│       │   ├── stats_generator.py
│       │   └── splitter.py       # split_dataset, SplitConfig, SplitResult
│       └── utils/
│           └── viz_utils.py      # bgr_to_rgb, draw_boxes
├── reports/
│   ├── figures/                  # Per-dataset + cross-dataset PNG+SVG figures
│   │   ├── dataset_size_comparison.png/.svg
│   │   ├── dataset_A/
│   │   │   └── * (22 figure types)
│   │   ├── dataset_B/
│   │   │   └── * (22 figure types)
│   │   ├── dataset_A_eda_report.md
│   │   └── dataset_B_eda_report.md
│   ├── preprocessing_stats/      # Pre/post stats CSVs
│   └── eda/                      # Inspection summary CSV
├── logs/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_preprocessing.ipynb
└── tests/
```

## Key Design Decisions

### Annotation Loading
- Single entry point: `load_dataset_annotations(spec: DatasetSpec) -> list[ImageAnnotation]`
- Supports Pascal VOC XML, YOLO .txt, CSV via `dataset.annotation_format` in YAML
- `BoundingBox` has properties: `width`, `height`, `area`, `center`

### EDA Figures (22 types)
1. `dataset_size_comparison` (cross-dataset bar chart)
2. `resolution_histograms`
3. `width_distribution`
4. `height_distribution`
5. `aspect_ratio_distribution`
6. `brightness_histogram`
7. `contrast_histogram`
8. `blur_estimation`
9. `sharpness_distribution`
10. `entropy_distribution`
11. `bbox_size_distribution` (combined)
12. `bbox_width_distribution`
13. `bbox_height_distribution`
14. `bbox_area_distribution`
15. `bbox_position_heatmap`
16. `class_distribution`
17. `example_images`
18. `random_samples`
19. `annotated_samples`
20. `color_distribution`
21. `duplicate_visualization`
22. `outlier_visualization`

All saved at **300 DPI** as `PNG + SVG` pairs.

### Figure Saving Convention
```python
def save_figure(fig: plt.Figure, output_dir: Path, name: str, dpi: int = 300):
    fig.savefig(output_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
```

### Validation
- `dataset_size_comparison` is cross-dataset — exclude from per-dataset file existence checks
- Validate all 22 (or 21 per-dataset) figure types exist as both `.png` and `.svg`
- `generate_all_eda_figures()` should handle missing optional args gracefully (empty list for missing annotations, empty list for missing duplicate pairs)

### Image I/O
- Always use `safe_read_image(path)` which does `cv2.imdecode(np.fromfile(path, np.uint8), ...)` for unicode path support
- Matplotlib must use `Agg` backend (no GUI)

### Configuration
- `pipeline_config.yaml`: paths (relative to project root), blur_threshold, duplicate_hash_threshold
- `datasets.yaml`: list of datasets with `name`, `root`, `annotation_format` (voc_xml/yolo/csv), `image_subdir`, `annotation_subdir`, optional `class_map`
- `preprocessing_config.yaml`: ordered steps list with `enabled` flag + `params` dict, split ratios, seed

### Preprocessing Pipeline
- Steps registered in `STEP_REGISTRY` dict mapping step name → function
- `PreprocessingPipeline` chains enabled steps in config order
- Each step receives `(image, **params)` and returns `np.ndarray`
- `letterbox` auto-fills `target_size` from top-level config
- All transforms write to `data/processed/preprocessed/<dataset_name>/`

### Splitting
- `split_dataset()` creates symlinks or copies files
- Stratification by class when `stratify_by_class: true`
- YOLO-style output: `<split>/images/`, `<split>/labels/`, `<split>/dataset.yaml`

### Notebooks
- Both must be valid `.ipynb` files using `nbformat`
- `01_eda.ipynb`: full EDA walkthrough, inline figure previews, validation
- `02_preprocessing.ipynb`: per-transform demo, before/after, stats, split
- No notebook execution required for pipeline operation

## Common Pitfalls / Gotchas

1. **`dataset_size_comparison` is cross-dataset** — validation counts must exclude it from per-dataset checks
2. **Variable name mismatches** — e.g., `entropy` vs `entropies` in report.py (already fixed)
3. **Unicode paths** — always use `safe_read_image`, never `cv2.imread(path)`
4. **Missing dependencies** — check `matplotlib`, `pandas`, `PyYAML`, `tqdm`, `opencv-python`, `numpy`, `Pillow` are installed
5. **Dataset_A `annotation_format`** is `voc_xml` with `image_subdir: Pics`, `annotation_subdir: annotations` — not `csv`
6. **Dataset_B** has no subdirs, uses `image_subdir: null`, `annotation_subdir: null`
7. **`cv2.imwrite` + unicode paths** — use `cv2.imencode(...)[1].tofile(str(path))` instead
8. **All outputs must be relative to project root** — don't hardcode absolute paths

## Testing
- Run `python -m pytest tests/` after verifying the codebase has tests
- Run each script independently from project root: `python scripts/run_*.py`
- Validate figure counts after EDA
- Run full pipeline: `python scripts/run_full_pipeline.py`
