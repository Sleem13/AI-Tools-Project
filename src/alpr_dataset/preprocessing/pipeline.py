"""Part 5 — chains configured preprocessing steps and applies them to a dataset,
saving processed images plus before/after comparison figures.
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
from tqdm import tqdm

from alpr_dataset.config import PreprocessingConfig
from alpr_dataset.io_utils import ensure_dir, safe_read_image
from alpr_dataset.preprocessing import transforms as T

logger = logging.getLogger("alpr_dataset")

# Maps config step names -> callables. Keeping this explicit (rather than
# getattr-based dispatch) avoids accidentally exposing unrelated functions
# to YAML-driven configuration.
STEP_REGISTRY = {
    "resize": T.resize,
    "letterbox": T.letterbox,
    "pad": T.pad,
    "denoise": T.denoise,
    "histogram_equalization": T.histogram_equalization,
    "clahe": T.clahe,
    "gamma_correction": T.gamma_correction,
    "brightness_normalization": T.brightness_normalization,
    "contrast_normalization": T.contrast_normalization,
    "gaussian_blur": T.gaussian_blur,
    "median_filter": T.median_filter,
    "bilateral_filter": T.bilateral_filter,
    "sharpen": T.sharpen,
    "perspective_correction": T.perspective_correction_hook,
    "rotation_correction": T.rotation_correction_hook,
}


class PreprocessingPipeline:
    """Applies a configured, ordered sequence of transforms to images."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self._resolved_steps = self._resolve_steps()

    def _resolve_steps(self) -> list[tuple[str, callable, dict]]:
        resolved = []
        for step in self.config.steps:
            if not step.enabled:
                continue
            fn = STEP_REGISTRY.get(step.name)
            if fn is None:
                logger.warning("Unknown preprocessing step '%s' — skipping", step.name)
                continue
            params = dict(step.params)
            if step.name in ("resize", "letterbox") and "target_size" not in params:
                params["target_size"] = self.config.target_size
            resolved.append((step.name, fn, params))
        return resolved

    def apply(self, image: np.ndarray) -> np.ndarray:
        out = image
        for name, fn, params in self._resolved_steps:
            try:
                out = fn(out, **params)
            except Exception as exc:
                logger.warning("Preprocessing step '%s' failed: %s — skipping step", name, exc)
        return out

    def run_on_dataset(
        self,
        image_paths: list[Path],
        output_dir: Path,
        dataset_name: str,
        comparisons_dir: Path | None = None,
    ) -> list[Path]:
        """Apply the pipeline to every image, saving processed outputs.

        If `comparisons_dir` is given, also saves before/after figures for
        `config.before_after_sample_count` randomly chosen images.
        """
        output_dir = ensure_dir(output_dir)
        processed_paths: list[Path] = []

        comparison_sample: set[Path] = set()
        if comparisons_dir and self.config.save_before_after:
            rng = random.Random(42)
            comparison_sample = set(
                rng.sample(image_paths, min(self.config.before_after_sample_count, len(image_paths)))
            )
            ensure_dir(comparisons_dir)

        for path in tqdm(image_paths, desc=f"[{dataset_name}] preprocessing", leave=False):
            img = safe_read_image(path)
            if img is None:
                logger.warning("Skipping unreadable image during preprocessing: %s", path)
                continue
            processed = self.apply(img)

            out_path = output_dir / f"{dataset_name}__{path.stem}.jpg"
            cv2.imwrite(str(out_path), processed)
            processed_paths.append(out_path)

            if path in comparison_sample:
                self._save_before_after(img, processed, comparisons_dir, f"{dataset_name}__{path.stem}")

        logger.info(
            "Preprocessed %d/%d images for '%s' -> %s",
            len(processed_paths),
            len(image_paths),
            dataset_name,
            output_dir,
        )
        return processed_paths

    @staticmethod
    def _save_before_after(before: np.ndarray, after: np.ndarray, out_dir: Path, name: str) -> None:
        def to_rgb(img: np.ndarray) -> np.ndarray:
            if img.ndim == 2:
                return img
            return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(to_rgb(before), cmap="gray" if before.ndim == 2 else None)
        axes[0].set_title("Before")
        axes[0].axis("off")
        axes[1].imshow(to_rgb(after), cmap="gray" if after.ndim == 2 else None)
        axes[1].set_title("After")
        axes[1].axis("off")
        fig.suptitle(name)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}_before_after.png", dpi=130)
        plt.close(fig)
