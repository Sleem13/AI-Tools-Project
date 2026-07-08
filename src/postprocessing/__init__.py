"""Post-processing for raw OCR output — plate formatting and validation.

Provides:
  - format_plate: apply Egyptian plate format rules to raw text
  - validate_plate: check if a plate string matches expected patterns
"""

from src.postprocessing.plate_formatter import format_plate, validate_plate

__all__ = ["format_plate", "validate_plate"]
