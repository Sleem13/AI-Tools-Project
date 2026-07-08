"""Egyptian license plate formatting and validation.

Egyptian plates follow patterns like:
  - Private cars: 1234 ABC  (numbers then letters)
  - Taxis: etc.

This module provides regex-based formatting and confidence-based
filtering to clean up raw OCR output.
"""

from __future__ import annotations

import re


def format_plate(raw_text: str) -> str:
    """Apply basic Egyptian plate formatting to raw OCR output.

    Strips non-alphanumeric characters and applies common formatting
    rules (e.g. uppercase).

    Args:
        raw_text: Raw string from OCR model.

    Returns:
        Formatted plate string, or empty string if unreadable.
    """
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_text).strip().upper()
    return cleaned


def validate_plate(text: str, min_length: int = 3, max_length: int = 10) -> bool:
    """Check whether a plate string is structurally valid.

    Args:
        text: Plate string to validate.
        min_length: Minimum acceptable length.
        max_length: Maximum acceptable length.

    Returns:
        True if the plate passes basic sanity checks.
    """
    if not text:
        return False
    if not (min_length <= len(text) <= max_length):
        return False
    if not re.search(r"[A-Z]", text):
        return False
    if not re.search(r"[0-9]", text):
        return False
    return True
