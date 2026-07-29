"""Import human review JSON into a retraining-friendly feedback folder."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "reviewed" / "human_feedback"


def import_review_feedback(review_json: Path, output_dir: Path = DEFAULT_OUTPUT) -> Path:
    payload = json.loads(review_json.read_text(encoding="utf-8"))
    media_stem = _safe_stem(str(payload.get("media") or review_json.stem))
    session_dir = output_dir / media_stem
    crops_dir = session_dir / "plate_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    detections_by_id = {item.get("id"): item for item in payload.get("detections", []) if item.get("id")}
    review_items = payload.get("review_items") or _build_review_items(payload)
    rows: list[dict[str, Any]] = []

    for index, item in enumerate(review_items):
        detection_id = str(item.get("id") or f"plate-{index + 1}")
        detection = detections_by_id.get(detection_id, {})
        crop_path = _write_crop(detection.get("plate_crop"), crops_dir / f"{detection_id}.jpg")
        corrected_text = (item.get("corrected_text") or "").strip()
        model_text = item.get("model_text") or detection.get("formatted_text") or detection.get("plate_text") or ""
        final_text = corrected_text or item.get("final_text") or model_text
        rows.append(
            {
                "media": payload.get("media") or "",
                "detection_id": detection_id,
                "decision": item.get("decision") or "",
                "model_text": model_text,
                "corrected_text": corrected_text,
                "final_text": final_text,
                "correction_changed": bool(corrected_text and corrected_text != model_text),
                "bbox": json.dumps(item.get("bbox") or detection.get("bbox") or [], ensure_ascii=False),
                "confidence": item.get("confidence") or detection.get("confidence") or "",
                "text_source": item.get("text_source") or detection.get("text_source") or "",
                "character_count": item.get("character_count") or len(detection.get("characters", [])),
                "crop_path": str(crop_path.relative_to(session_dir)) if crop_path else "",
            }
        )

    manifest = {
        "source_review": str(review_json),
        "media": payload.get("media"),
        "reviewed_at": payload.get("reviewed_at"),
        "items": rows,
        "note": "Corrected text is review metadata. YOLO character fine-tuning still needs character boxes for any missing or changed glyphs.",
    }
    (session_dir / "review_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with (session_dir / "plate_text_corrections.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["media"])
        writer.writeheader()
        writer.writerows(rows)
    return session_dir


def _build_review_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    reviews = payload.get("reviews", {})
    corrections = payload.get("corrections", {})
    items = []
    for index, detection in enumerate(payload.get("detections", [])):
        detection_id = detection.get("id") or f"plate-{index + 1}"
        corrected_text = (corrections.get(detection_id) or "").strip()
        model_text = detection.get("formatted_text") or detection.get("plate_text") or ""
        items.append(
            {
                "id": detection_id,
                "index": index,
                "decision": reviews.get(detection_id),
                "model_text": model_text,
                "raw_model_text": detection.get("plate_text") or "",
                "corrected_text": corrected_text or None,
                "final_text": corrected_text or model_text,
                "correction_changed": bool(corrected_text and corrected_text != model_text),
                "bbox": detection.get("bbox") or [],
                "confidence": detection.get("confidence"),
                "text_source": detection.get("text_source"),
                "character_count": len(detection.get("characters", [])),
            }
        )
    return items


def _write_crop(data_url: str | None, output_path: Path) -> Path | None:
    if not data_url or "," not in data_url:
        return None
    _, encoded = data_url.split(",", 1)
    try:
        output_path.write_bytes(base64.b64decode(encoded))
    except ValueError:
        return None
    return output_path


def _safe_stem(value: str) -> str:
    stem = Path(value).stem or "review"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "review"


def main() -> None:
    parser = argparse.ArgumentParser(description="Import ALPR human review JSON for retraining/annotation.")
    parser.add_argument("review_json", type=Path, help="Exported review JSON from the Vite human review lab.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Destination feedback dataset root.")
    args = parser.parse_args()
    session_dir = import_review_feedback(args.review_json, args.output_dir)
    print(session_dir)


if __name__ == "__main__":
    main()
