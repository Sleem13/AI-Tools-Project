"""Unit tests for preprocessing transforms and pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from alpr_dataset.config import PreprocessingConfig, PreprocessingStepConfig
from alpr_dataset.preprocessing import transforms as T
from alpr_dataset.preprocessing.pipeline import PreprocessingPipeline


@pytest.fixture()
def gray_image() -> np.ndarray:
    """Synthetic 64x128 grayscale image."""
    img = np.zeros((64, 128), dtype=np.uint8)
    img[20:40, 30:90] = 200
    return img


@pytest.fixture()
def bgr_image() -> np.ndarray:
    """Synthetic 64x128 BGR image."""
    img = np.zeros((64, 128, 3), dtype=np.uint8)
    img[20:40, 30:90] = (100, 150, 200)
    return img


class TestResize:
    def test_output_shape(self, bgr_image: np.ndarray) -> None:
        result = T.resize(bgr_image, (64, 32))
        assert result.shape == (32, 64, 3)

    def test_grayscale(self, gray_image: np.ndarray) -> None:
        result = T.resize(gray_image, (32, 16))
        assert result.shape == (16, 32)


class TestLetterbox:
    def test_preserves_aspect_ratio(self, bgr_image: np.ndarray) -> None:
        result = T.letterbox(bgr_image, (200, 200))
        assert result.shape == (200, 200, 3)
        # Non-content pixels should be the padding color (114, 114, 114)
        assert result[0, 0, 0] == 114


class TestPad:
    def test_output_size(self, bgr_image: np.ndarray) -> None:
        result = T.pad(bgr_image, 10)
        assert result.shape == (84, 148, 3)


class TestDenoise:
    def test_output_same_shape(self, gray_image: np.ndarray) -> None:
        result = T.denoise(gray_image, strength=5.0)
        assert result.shape == gray_image.shape

    def test_color_image(self, bgr_image: np.ndarray) -> None:
        result = T.denoise(bgr_image, strength=5.0)
        assert result.shape == bgr_image.shape


class TestHistogramEqualization:
    def test_grayscale(self, gray_image: np.ndarray) -> None:
        result = T.histogram_equalization(gray_image)
        assert result.shape == gray_image.shape

    def test_color(self, bgr_image: np.ndarray) -> None:
        result = T.histogram_equalization(bgr_image)
        assert result.shape == bgr_image.shape


class TestClahe:
    def test_grayscale(self, gray_image: np.ndarray) -> None:
        result = T.clahe(gray_image)
        assert result.shape == gray_image.shape

    def test_color(self, bgr_image: np.ndarray) -> None:
        result = T.clahe(bgr_image)
        assert result.shape == bgr_image.shape


class TestGammaCorrection:
    def test_identity_gamma(self, bgr_image: np.ndarray) -> None:
        result = T.gamma_correction(bgr_image, gamma=1.0)
        np.testing.assert_array_equal(result, bgr_image)

    def test_brighten(self, bgr_image: np.ndarray) -> None:
        result = T.gamma_correction(bgr_image, gamma=2.0)
        assert result.mean() >= bgr_image.mean()


class TestBrightnessNormalization:
    def test_target_mean(self, bgr_image: np.ndarray) -> None:
        result = T.brightness_normalization(bgr_image, target_mean=128.0)
        assert abs(float(result.mean()) - 128.0) < 5.0


class TestContrastNormalization:
    def test_output_shape(self, bgr_image: np.ndarray) -> None:
        result = T.contrast_normalization(bgr_image, target_std=60.0)
        assert result.shape == bgr_image.shape


class TestGaussianBlur:
    def test_output_shape(self, bgr_image: np.ndarray) -> None:
        result = T.gaussian_blur(bgr_image, kernel_size=5)
        assert result.shape == bgr_image.shape


class TestMedianFilter:
    def test_output_shape(self, bgr_image: np.ndarray) -> None:
        result = T.median_filter(bgr_image, kernel_size=3)
        assert result.shape == bgr_image.shape


class TestSharpen:
    def test_output_shape(self, bgr_image: np.ndarray) -> None:
        result = T.sharpen(bgr_image, amount=0.5)
        assert result.shape == bgr_image.shape


class TestPerspectiveCorrectionHook:
    def test_noop_without_corners(self, bgr_image: np.ndarray) -> None:
        result = T.perspective_correction_hook(bgr_image)
        np.testing.assert_array_equal(result, bgr_image)

    def test_noop_with_wrong_count(self, bgr_image: np.ndarray) -> None:
        corners = np.array([[0, 0], [100, 0], [100, 50]])
        result = T.perspective_correction_hook(bgr_image, corners)
        np.testing.assert_array_equal(result, bgr_image)


class TestRotationCorrectionHook:
    def test_noop_with_zero_angle(self, bgr_image: np.ndarray) -> None:
        result = T.rotation_correction_hook(bgr_image, angle=0.0)
        np.testing.assert_array_equal(result, bgr_image)

    def test_rotates_nonzero_angle(self, bgr_image: np.ndarray) -> None:
        result = T.rotation_correction_hook(bgr_image, angle=15.0)
        assert result.shape == bgr_image.shape


class TestPreprocessingPipeline:
    def test_apply_empty_pipeline(self, bgr_image: np.ndarray) -> None:
        cfg = PreprocessingConfig(target_size=(640, 640), steps=[])
        pipeline = PreprocessingPipeline(cfg)
        result = pipeline.apply(bgr_image)
        np.testing.assert_array_equal(result, bgr_image)

    def test_apply_clahe(self, bgr_image: np.ndarray) -> None:
        cfg = PreprocessingConfig(
            target_size=(640, 640),
            steps=[PreprocessingStepConfig(name="clahe", enabled=True, params={"clip_limit": 2.0})],
        )
        pipeline = PreprocessingPipeline(cfg)
        result = pipeline.apply(bgr_image)
        assert result.shape == bgr_image.shape
        # CLAHE should change pixel values
        assert not np.array_equal(result, bgr_image)

    def test_disabled_steps_are_skipped(self, bgr_image: np.ndarray) -> None:
        cfg = PreprocessingConfig(
            target_size=(640, 640),
            steps=[PreprocessingStepConfig(name="clahe", enabled=False)],
        )
        pipeline = PreprocessingPipeline(cfg)
        result = pipeline.apply(bgr_image)
        np.testing.assert_array_equal(result, bgr_image)

    def test_chained_transforms(self, bgr_image: np.ndarray) -> None:
        cfg = PreprocessingConfig(
            target_size=(128, 64),
            steps=[
                PreprocessingStepConfig(name="resize", params={"target_size": (128, 64)}),
                PreprocessingStepConfig(name="clahe"),
                PreprocessingStepConfig(name="sharpen", params={"amount": 0.5}),
            ],
        )
        pipeline = PreprocessingPipeline(cfg)
        result = pipeline.apply(bgr_image)
        assert result.shape == (64, 128, 3)

    def test_run_on_dataset(self, bgr_image: np.ndarray, tmp_path: Path) -> None:
        # Create test images
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for i in range(3):
            cv2.imwrite(str(img_dir / f"img_{i}.jpg"), bgr_image)

        cfg = PreprocessingConfig(
            target_size=(128, 64),
            steps=[PreprocessingStepConfig(name="resize", params={"target_size": (128, 64)})],
            save_before_after=False,
        )
        pipeline = PreprocessingPipeline(cfg)
        output_dir = tmp_path / "output"
        image_paths = sorted(img_dir.glob("*.jpg"))
        result = pipeline.run_on_dataset(image_paths, output_dir, "test_ds")
        assert len(result) == 3
        for p in result:
            assert p.exists()
