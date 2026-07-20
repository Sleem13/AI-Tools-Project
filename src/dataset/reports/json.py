from __future__ import annotations

import json
from pathlib import Path

from ..inspector.validator import ValidationSummary


class JSONReportGenerator:
    @staticmethod
    def generate_quality_report(
        validation_summaries: list[tuple[str, ValidationSummary]],
        output_path: Path,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data: dict = {
            "generated_by": "Dataset Inspector",
            "datasets": {},
        }

        for dataset_name, summary in validation_summaries:
            data["datasets"][dataset_name] = summary.to_dict()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        return output_path
