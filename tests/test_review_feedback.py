from __future__ import annotations

import base64
import json
from pathlib import Path

from scripts.import_review_feedback import import_review_feedback


def test_import_review_feedback_writes_corrections_and_crops(tmp_path: Path) -> None:
    crop_data = "data:image/jpeg;base64," + base64.b64encode(b"fake-jpeg").decode("ascii")
    review_path = tmp_path / "ducati-review.json"
    review_path.write_text(
        json.dumps(
            {
                "media": "ducati.webp",
                "reviewed_at": "2026-07-29T00:00:00.000Z",
                "detections": [
                    {
                        "id": "plate-2",
                        "bbox": [692, 376, 726, 430],
                        "confidence": 0.82,
                        "plate_text": "جنا / 35",
                        "formatted_text": "جنا 35",
                        "text_source": "character_detector",
                        "characters": [{"glyph": "ج"}],
                        "plate_crop": crop_data,
                    }
                ],
                "reviews": {"plate-2": "needs_review"},
                "corrections": {"plate-2": "حنا 5728"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    session_dir = import_review_feedback(review_path, tmp_path / "feedback")

    manifest = json.loads((session_dir / "review_manifest.json").read_text(encoding="utf-8"))
    assert manifest["items"][0]["model_text"] == "جنا 35"
    assert manifest["items"][0]["corrected_text"] == "حنا 5728"
    assert manifest["items"][0]["correction_changed"] is True
    assert (session_dir / "plate_crops" / "plate-2.jpg").read_bytes() == b"fake-jpeg"
    assert "حنا 5728" in (session_dir / "plate_text_corrections.csv").read_text(encoding="utf-8-sig")
