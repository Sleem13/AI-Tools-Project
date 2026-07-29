from __future__ import annotations

import numpy as np

from src.evaluation.pipeline import ALPRPipeline


class FakeDetector:
    def crop_plates(self, image: np.ndarray, min_confidence: float = 0.25):
        crop = image[:2, :2].copy()
        return [(crop, 0.75)]


class FakeReader:
    def read_plate(self, crop: np.ndarray) -> str:
        return "abc123"


def test_pipeline_results_include_confidence() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    pipeline = ALPRPipeline(detector=FakeDetector(), reader=FakeReader())

    results = pipeline.run(image)

    assert results == [
        {
            "detection_confidence": 0.75,
            "plate_text": "abc123",
            "formatted_text": "ABC 123",
            "confidence": 1.0,
        }
    ]


def test_pipeline_filters_by_min_confidence() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    pipeline = ALPRPipeline(detector=FakeDetector(), reader=FakeReader(), min_confidence=0.8)

    assert pipeline.run(image) == []
