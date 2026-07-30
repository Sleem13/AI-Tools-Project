"""Part 2 — publication-quality EDA figures.

Every plotting function saves both a PNG (raster, for reports/slides) and
an SVG (vector, for papers) into the given output directory via
`save_figure`. Matplotlib only (no seaborn), per project style rules.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from alpr_dataset.annotations.schema import ImageAnnotation
from alpr_dataset.inspection.image_stats import ImageStats
from alpr_dataset.io_utils import ensure_dir, safe_read_image

logger = logging.getLogger("alpr_dataset")

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


_FIGSIZE = (7, 4.5)


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir = ensure_dir(output_dir)
    png_path = output_dir / f"{name}.png"
    svg_path = output_dir / f"{name}.svg"
    fig.tight_layout()
    fig.savefig(png_path, format="png")
    fig.savefig(svg_path, format="svg")
    plt.close(fig)
    logger.info("Saved figure: %s (.png/.svg)", name)


# --------------------------------------------------------------------------- #
# Dataset-level comparisons
# --------------------------------------------------------------------------- #
def plot_dataset_size_comparison(counts: dict[str, int], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    names, values = list(counts.keys()), list(counts.values())
    bars = ax.bar(names, values, color="#3d5a80", edgecolor="white", linewidth=0.6)
    ax.set_ylabel("Number of images")
    ax.set_title("Dataset Size Comparison")
    for bar, v in zip(bars, values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            str(v),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    save_figure(fig, output_dir, "dataset_size_comparison")


# --------------------------------------------------------------------------- #
# Resolution / geometry
# --------------------------------------------------------------------------- #
def plot_resolution_histograms(widths: list[int], heights: list[int], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(widths, bins=40, color="#3d5a80", edgecolor="white")
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Width Distribution")
    axes[1].hist(heights, bins=40, color="#ee6c4d", edgecolor="white")
    axes[1].set_xlabel("Height (px)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Height Distribution")
    save_figure(fig, output_dir, "resolution_histograms")


def plot_width_distribution(widths: list[int], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.hist(widths, bins=40, color="#3d5a80", edgecolor="white")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Frequency")
    ax.set_title("Image Width Distribution")
    save_figure(fig, output_dir, "width_distribution")


def plot_height_distribution(heights: list[int], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.hist(heights, bins=40, color="#ee6c4d", edgecolor="white")
    ax.set_xlabel("Height (px)")
    ax.set_ylabel("Frequency")
    ax.set_title("Image Height Distribution")
    save_figure(fig, output_dir, "height_distribution")


def plot_aspect_ratio_distribution(ratios: list[float], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.hist(ratios, bins=40, color="#98c1d9", edgecolor="white")
    ax.set_xlabel("Aspect ratio (width / height)")
    ax.set_ylabel("Frequency")
    ax.set_title("Aspect Ratio Distribution")
    save_figure(fig, output_dir, "aspect_ratio_distribution")


# --------------------------------------------------------------------------- #
# Bounding boxes
# --------------------------------------------------------------------------- #
def _get_bbox_widths(annotations: list[ImageAnnotation]) -> list[float]:
    return [b.width for a in annotations for b in a.boxes if b.width > 0]


def _get_bbox_heights(annotations: list[ImageAnnotation]) -> list[float]:
    return [b.height for a in annotations for b in a.boxes if b.height > 0]


def _get_bbox_areas(annotations: list[ImageAnnotation]) -> list[float]:
    return [b.area for a in annotations for b in a.boxes if b.area > 0]


def plot_bbox_width_distribution(annotations: list[ImageAnnotation], output_dir: Path) -> None:
    widths = _get_bbox_widths(annotations)
    if not widths:
        logger.warning("No bounding boxes available for bbox width plot")
        return
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.hist(widths, bins=40, color="#3d5a80", edgecolor="white")
    ax.set_xlabel("Bounding box width (px)")
    ax.set_ylabel("Frequency")
    ax.set_title("Bounding Box Width Distribution")
    save_figure(fig, output_dir, "bbox_width_distribution")


def plot_bbox_height_distribution(annotations: list[ImageAnnotation], output_dir: Path) -> None:
    heights = _get_bbox_heights(annotations)
    if not heights:
        logger.warning("No bounding boxes available for bbox height plot")
        return
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.hist(heights, bins=40, color="#ee6c4d", edgecolor="white")
    ax.set_xlabel("Bounding box height (px)")
    ax.set_ylabel("Frequency")
    ax.set_title("Bounding Box Height Distribution")
    save_figure(fig, output_dir, "bbox_height_distribution")


def plot_bbox_area_distribution(annotations: list[ImageAnnotation], output_dir: Path) -> None:
    areas = _get_bbox_areas(annotations)
    if not areas:
        logger.warning("No bounding boxes available for bbox area plot")
        return
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.hist(areas, bins=40, color="#4ba36f", edgecolor="white")
    ax.set_xlabel("Bounding box area (px²)")
    ax.set_ylabel("Frequency")
    ax.set_title("Bounding Box Area Distribution")
    save_figure(fig, output_dir, "bbox_area_distribution")


def plot_bbox_size_distribution(annotations: list[ImageAnnotation], output_dir: Path) -> None:
    widths = _get_bbox_widths(annotations)
    heights = _get_bbox_heights(annotations)
    if not widths or not heights:
        logger.warning("No bounding boxes available for bbox size distribution plot")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(widths, bins=40, color="#3d5a80", edgecolor="white")
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Bounding Box Width")
    axes[1].hist(heights, bins=40, color="#ee6c4d", edgecolor="white")
    axes[1].set_xlabel("Height (px)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Bounding Box Height")
    save_figure(fig, output_dir, "bbox_size_distribution")


def plot_bbox_position_heatmap(
    annotations: list[ImageAnnotation], output_dir: Path, grid_size: int = 32
) -> None:
    heat = np.zeros((grid_size, grid_size), dtype=np.float64)
    for ann in annotations:
        if ann.image_width <= 0 or ann.image_height <= 0:
            continue
        for box in ann.boxes:
            cx, cy = box.center
            gx = min(int(cx / ann.image_width * grid_size), grid_size - 1)
            gy = min(int(cy / ann.image_height * grid_size), grid_size - 1)
            if 0 <= gx < grid_size and 0 <= gy < grid_size:
                heat[gy, gx] += 1
    if heat.sum() == 0:
        logger.warning("No bounding boxes available for position heatmap")
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(heat, cmap="inferno", origin="upper")
    ax.set_title("Bounding Box Position Heatmap (normalized image space)")
    ax.set_xlabel("Normalised x →")
    ax.set_ylabel("Normalised y →")
    fig.colorbar(im, ax=ax, label="Box centre count")
    save_figure(fig, output_dir, "bbox_position_heatmap")


def plot_class_distribution(annotations: list[ImageAnnotation], output_dir: Path) -> None:
    from collections import Counter

    counts = Counter(b.class_name or str(b.class_id) for a in annotations for b in a.boxes)
    if not counts:
        logger.warning("No class labels available for class distribution plot")
        return
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    names, values = zip(*sorted(counts.items(), key=lambda x: -x[1]), strict=False)
    ax.bar(names, values, color="#3d5a80", edgecolor="white", linewidth=0.6)
    ax.set_ylabel("Frequency")
    ax.set_title("Class Distribution")
    ax.tick_params(axis="x", rotation=45)
    save_figure(fig, output_dir, "class_distribution")


# --------------------------------------------------------------------------- #
# Image galleries
# --------------------------------------------------------------------------- #
def _grid_of_images(paths: list[Path], title: str, n: int = 9) -> plt.Figure | None:
    n = min(n, len(paths))
    if n == 0:
        return None
    cols = 3
    rows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            img = safe_read_image(paths[i])
            if img is None:
                continue
            if img.ndim == 3 and img.shape[2] >= 3:
                img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.set_title(paths[i].name, fontsize=7)
    fig.suptitle(title, fontsize=14)
    return fig


def plot_example_images(image_paths: list[Path], output_dir: Path, n: int = 9) -> None:
    fig = _grid_of_images(image_paths[:n], "Example Images", n)
    if fig is None:
        logger.warning("No images available for example-images plot")
        return
    save_figure(fig, output_dir, "example_images")


def plot_random_samples(image_paths: list[Path], output_dir: Path, n: int = 9, seed: int = 42) -> None:
    rng = random.Random(seed)
    sample = rng.sample(image_paths, min(n, len(image_paths)))
    fig = _grid_of_images(sample, "Random Samples", n)
    if fig is None:
        return
    save_figure(fig, output_dir, "random_samples")


def plot_annotated_samples(
    annotations: list[ImageAnnotation], output_dir: Path, n: int = 9, seed: int = 42
) -> None:
    rng = random.Random(seed)
    with_boxes = [a for a in annotations if a.boxes]
    if not with_boxes:
        logger.warning("No annotated images available for annotated-samples plot")
        return
    sample = rng.sample(with_boxes, min(n, len(with_boxes)))

    cols = 3
    rows = max(1, (len(sample) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= len(sample):
            continue
        ann = sample[i]
        img = safe_read_image(ann.image_path)
        if img is None:
            continue
        if img.ndim == 3 and img.shape[2] >= 3:
            img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        for box in ann.boxes:
            rect = plt.Rectangle(
                (box.x_min, box.y_min),
                box.width,
                box.height,
                fill=False,
                edgecolor="#ee6c4d",
                linewidth=2,
            )
            ax.add_patch(rect)
        ax.set_title(ann.image_path.name, fontsize=7)
    fig.suptitle("Annotated Samples", fontsize=14)
    save_figure(fig, output_dir, "annotated_samples")


# --------------------------------------------------------------------------- #
# Photometric quality
# --------------------------------------------------------------------------- #
def plot_brightness_histogram(stats: list[ImageStats], output_dir: Path) -> None:
    values = [s.brightness_mean for s in stats if not s.is_corrupted]
    if not values:
        return
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.hist(values, bins=40, color="#f4d35e", edgecolor="white")
    ax.set_xlabel("Mean brightness (0 – 255)")
    ax.set_ylabel("Frequency")
    ax.set_title("Brightness Distribution")
    save_figure(fig, output_dir, "brightness_histogram")


def plot_contrast_histogram(stats: list[ImageStats], output_dir: Path) -> None:
    values = [s.contrast_std for s in stats if not s.is_corrupted]
    if not values:
        return
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.hist(values, bins=40, color="#ee6c4d", edgecolor="white")
    ax.set_xlabel("Pixel intensity standard deviation")
    ax.set_ylabel("Frequency")
    ax.set_title("Contrast Distribution")
    save_figure(fig, output_dir, "contrast_histogram")


def plot_blur_estimation(stats: list[ImageStats], output_dir: Path, threshold: float = 100.0) -> None:
    values = [s.blur_score for s in stats if not s.is_corrupted]
    if not values:
        return
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.hist(values, bins=40, color="#3d5a80", edgecolor="white")
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Blur threshold = {threshold}")
    ax.set_xlabel("Variance of Laplacian")
    ax.set_ylabel("Frequency")
    ax.set_title("Blur Estimation (lower values = blurrier)")
    ax.legend(fontsize=10)
    save_figure(fig, output_dir, "blur_estimation")


def plot_sharpness(stats: list[ImageStats], output_dir: Path) -> None:
    values = [s.sharpness_score for s in stats if not s.is_corrupted]
    if not values:
        return
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.hist(values, bins=40, color="#98c1d9", edgecolor="white")
    ax.set_xlabel("Mean Sobel gradient magnitude")
    ax.set_ylabel("Frequency")
    ax.set_title("Image Sharpness Distribution")
    save_figure(fig, output_dir, "sharpness_distribution")


def plot_entropy(stats: list[ImageStats], output_dir: Path) -> None:
    values = [s.entropy for s in stats if not s.is_corrupted]
    if not values:
        return
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.hist(values, bins=40, color="#293241", edgecolor="white")
    ax.set_xlabel("Shannon entropy (bits)")
    ax.set_ylabel("Frequency")
    ax.set_title("Image Entropy Distribution")
    save_figure(fig, output_dir, "entropy_distribution")


def plot_color_distribution(image_paths: list[Path], output_dir: Path, sample_size: int = 200) -> None:
    rng = random.Random(42)
    sample = rng.sample(image_paths, min(sample_size, len(image_paths)))
    channel_hists = {"B": np.zeros(256), "G": np.zeros(256), "R": np.zeros(256)}
    for p in sample:
        img = safe_read_image(p)
        if img is None or img.ndim != 3 or img.shape[2] < 3:
            continue
        for i, ch in enumerate(("B", "G", "R")):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256]).flatten()
            channel_hists[ch] += hist
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    for ch, color in zip(("B", "G", "R"), ("#3d5a80", "#4ba36f", "#ee6c4d"), strict=False):
        ax.plot(channel_hists[ch], color=color, label=ch, linewidth=1.2)
    ax.set_title(f"Aggregate Colour Channel Distribution (n={len(sample)} sampled)")
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Aggregate pixel count")
    ax.legend(fontsize=10)
    save_figure(fig, output_dir, "color_distribution")


# --------------------------------------------------------------------------- #
# Duplicates & outliers
# --------------------------------------------------------------------------- #
def plot_duplicate_visualization(
    duplicate_pairs: list[tuple[Path, Path, int]], output_dir: Path, n: int = 6
) -> None:
    if not duplicate_pairs:
        logger.warning("No duplicate pairs to visualise")
        return
    pairs = duplicate_pairs[:n]
    fig, axes = plt.subplots(len(pairs), 2, figsize=(6, 3 * len(pairs)))
    axes = np.atleast_2d(axes)
    for i, (path_a, path_b, distance) in enumerate(pairs):
        for j, p in enumerate((path_a, path_b)):
            ax = axes[i, j]
            ax.axis("off")
            img = safe_read_image(p)
            if img is None:
                continue
            if img.ndim == 3 and img.shape[2] >= 3:
                img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.set_title(f"{p.name}\n(hamming={distance})" if j == 1 else p.name, fontsize=7)
    fig.suptitle("Near-Duplicate Pairs", fontsize=14)
    save_figure(fig, output_dir, "duplicate_visualization")


def plot_outlier_visualization(stats: list[ImageStats], output_dir: Path, z_thresh: float = 3.0) -> None:
    """Scatter of resolution vs. file size, highlighting statistical outliers."""
    valid = [s for s in stats if not s.is_corrupted]
    if not valid:
        return
    resolutions = np.array([s.width * s.height for s in valid])
    sizes = np.array([s.file_size_bytes for s in valid])

    def _z(x: np.ndarray) -> np.ndarray:
        std = x.std() or 1.0
        return (x - x.mean()) / std

    z_res, z_size = _z(resolutions), _z(sizes)
    is_outlier = (np.abs(z_res) > z_thresh) | (np.abs(z_size) > z_thresh)

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.scatter(
        resolutions[~is_outlier],
        sizes[~is_outlier],
        s=15,
        alpha=0.5,
        color="#3d5a80",
        label=f"Normal ({int((~is_outlier).sum())})",
    )
    ax.scatter(
        resolutions[is_outlier],
        sizes[is_outlier],
        s=30,
        color="#ee6c4d",
        edgecolor="black",
        linewidth=0.5,
        label=f"Outlier ({int(is_outlier.sum())})",
    )
    ax.set_xlabel("Resolution (width × height, px)")
    ax.set_ylabel("File size (bytes)")
    ax.set_title(f"Outlier Detection (|z| > {z_thresh})")
    ax.legend(fontsize=10)
    save_figure(fig, output_dir, "outlier_visualization")


# --------------------------------------------------------------------------- #
# Master entry point
# --------------------------------------------------------------------------- #
def generate_all_eda_figures(
    dataset_counts: dict[str, int],
    stats: list[ImageStats],
    annotations: list[ImageAnnotation],
    image_paths: list[Path],
    duplicate_pairs: list[tuple[Path, Path, int]],
    output_dir: Path,
    blur_threshold: float = 100.0,
) -> None:
    """Convenience entry point that runs every Part-2 figure in sequence."""
    valid = [s for s in stats if not s.is_corrupted]
    widths = [s.width for s in valid]
    heights = [s.height for s in valid]
    ratios = [w / h for w, h in zip(widths, heights, strict=False) if h > 0]

    plot_dataset_size_comparison(dataset_counts, output_dir)
    plot_resolution_histograms(widths, heights, output_dir)
    plot_width_distribution(widths, output_dir)
    plot_height_distribution(heights, output_dir)
    plot_aspect_ratio_distribution(ratios, output_dir)
    plot_bbox_size_distribution(annotations, output_dir)
    plot_bbox_width_distribution(annotations, output_dir)
    plot_bbox_height_distribution(annotations, output_dir)
    plot_bbox_area_distribution(annotations, output_dir)
    plot_bbox_position_heatmap(annotations, output_dir)
    plot_class_distribution(annotations, output_dir)
    plot_example_images(image_paths, output_dir)
    plot_random_samples(image_paths, output_dir)
    plot_annotated_samples(annotations, output_dir)
    plot_brightness_histogram(stats, output_dir)
    plot_contrast_histogram(stats, output_dir)
    plot_blur_estimation(stats, output_dir, threshold=blur_threshold)
    plot_sharpness(stats, output_dir)
    plot_entropy(stats, output_dir)
    plot_color_distribution(image_paths, output_dir)
    plot_duplicate_visualization(duplicate_pairs, output_dir)
    plot_outlier_visualization(stats, output_dir)
