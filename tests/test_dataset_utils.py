"""Unit tests for src/dataset utility modules."""

from __future__ import annotations

from pathlib import Path

from dataset.utils.filesystem import (
    should_ignore,
)
from dataset.utils.hashing import compute_md5
from dataset.utils.image import (
    has_supported_extension,
    is_zero_byte,
    validate_image,
)


class TestShouldIgnore:
    def test_hidden_file(self, tmp_path: Path) -> None:
        f = tmp_path / ".hidden"
        f.touch()
        assert should_ignore(f, ignore_hidden=True, ignore_cache=True) is True

    def test_normal_file(self, tmp_path: Path) -> None:
        f = tmp_path / "normal.txt"
        f.touch()
        assert should_ignore(f, ignore_hidden=True, ignore_cache=True) is False

    def test_cache_file(self, tmp_path: Path) -> None:
        f = tmp_path / "__pycache__"
        f.mkdir()
        assert should_ignore(f, ignore_hidden=True, ignore_cache=True) is True

    def test_ignore_hidden_disabled(self, tmp_path: Path) -> None:
        f = tmp_path / ".hidden"
        f.touch()
        assert should_ignore(f, ignore_hidden=False, ignore_cache=True) is False


class TestComputeMd5:
    def test_deterministic(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h1 = compute_md5(f)
        h2 = compute_md5(f)
        assert h1 == h2
        assert len(h1) == 32  # MD5 hex digest length

    def test_different_files_differ(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("content A")
        f2.write_text("content B")
        assert compute_md5(f1) != compute_md5(f2)


class TestIsZeroByte:
    def test_zero_byte(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        assert is_zero_byte(f) is True

    def test_non_zero(self, tmp_path: Path) -> None:
        f = tmp_path / "data.txt"
        f.write_text("x")
        assert is_zero_byte(f) is False


class TestHasSupportedExtension:
    def test_jpg(self, tmp_path: Path) -> None:
        f = tmp_path / "image.jpg"
        assert has_supported_extension(f) is True

    def test_unsupported(self, tmp_path: Path) -> None:
        f = tmp_path / "image.xyz"
        assert has_supported_extension(f) is False


class TestValidateImage:
    def test_nonexistent(self, tmp_path: Path) -> None:
        result = validate_image(tmp_path / "nope.jpg")
        assert result["exists"] is False
        assert result["error"] is not None

    def test_zero_byte(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.jpg"
        f.write_bytes(b"")
        result = validate_image(f)
        assert result["zero_byte"] is True

    def test_valid_image(self, tmp_image: Path) -> None:
        result = validate_image(tmp_image)
        assert result["exists"] is True
        assert result["corrupted"] is False
        assert result["readable"] is True
        assert result["size"] is not None
