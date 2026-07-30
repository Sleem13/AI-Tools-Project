"""Tests for API health model source reporting."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from api.main import app
from fastapi.testclient import TestClient


def test_health_reports_ocr_source(monkeypatch) -> None:
    import api.deps

    monkeypatch.setattr(api.deps, "get_detector", lambda: SimpleNamespace(character_detector=object()))
    monkeypatch.setattr(
        api.deps,
        "get_reader",
        lambda: SimpleNamespace(
            source="keras_crnn",
            weights_path=Path("models/weights/plate_ocr_weights.weights.h5"),
        ),
    )
    client = TestClient(app)

    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["models_loaded"] == {"detection": True, "character": True, "ocr": True}
    assert payload["model_sources"]["ocr"] == "keras_crnn"
    assert payload["model_sources"]["ocr_path"].endswith("plate_ocr_weights.weights.h5")
