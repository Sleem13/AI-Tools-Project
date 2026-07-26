"""Tests for Egyptian plate text formatting."""
# ruff: noqa: RUF001

from src.postprocessing.plate_formatter import format_plate, validate_plate


def test_formatter_preserves_arabic_characters_and_group_spacing() -> None:
    assert format_plate("  ا س   2 1  ") == "ا س 2 1"


def test_formatter_keeps_existing_latin_ocr_behavior() -> None:
    assert format_plate("abc-123") == "ABC123"


def test_validator_accepts_arabic_letters_with_digits() -> None:
    assert validate_plate("اس21")
