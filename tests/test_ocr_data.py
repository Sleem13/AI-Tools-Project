"""Unit tests for OCR data utilities (vocab, encode/decode)."""

from __future__ import annotations

import pytest

from alpr_dataset.annotations.schema import BoundingBox, ImageAnnotation
from alpr_dataset.config import SplitConfig
from alpr_dataset.splitting.splitter import stratified_split


class TestBuildVocab:
    def test_blank_token_at_index_zero(self) -> None:
        from src.ocr.data import build_vocab, CTC_BLANK

        vocab = build_vocab()
        assert vocab[CTC_BLANK] == 0

    def test_all_digits_present(self) -> None:
        from src.ocr.data import build_vocab

        vocab = build_vocab()
        for d in "0123456789":
            assert d in vocab

    def test_all_uppercase_present(self) -> None:
        from src.ocr.data import build_vocab

        vocab = build_vocab()
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            assert c in vocab

    def test_custom_chars(self) -> None:
        from src.ocr.data import build_vocab

        vocab = build_vocab("AB")
        assert "A" in vocab
        assert "B" in vocab
        assert "C" not in vocab


class TestEncodeDecode:
    def test_roundtrip(self) -> None:
        from src.ocr.data import build_vocab, encode_text, decode_text

        vocab = build_vocab()
        text = "ABC123"
        encoded = encode_text(text, vocab)
        decoded = decode_text(encoded, vocab)
        assert decoded == text

    def test_ignores_unknown_chars(self) -> None:
        from src.ocr.data import build_vocab, encode_text

        vocab = build_vocab("ABC")
        encoded = encode_text("AXBYC", vocab)
        assert encoded == [vocab["A"], vocab["B"], vocab["C"]]

    def test_decode_ignores_blank(self) -> None:
        from src.ocr.data import build_vocab, decode_text, CTC_BLANK

        vocab = build_vocab("AB")
        decoded = decode_text([CTC_BLANK, vocab["A"], CTC_BLANK, vocab["B"]], vocab)
        assert decoded == "AB"
