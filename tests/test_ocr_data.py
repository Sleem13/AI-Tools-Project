"""Unit tests for OCR data utilities (vocab, encode/decode)."""

from __future__ import annotations

import pytest
import numpy as np

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


def test_reader_preprocess_matches_training_width(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.ocr.inference import PlateReader

    captured_shape: tuple[int, ...] | None = None

    class FakeModel:
        def forward_for_ctc(self, tensor):
            nonlocal captured_shape
            captured_shape = tuple(tensor.shape)
            import torch

            return torch.zeros((1, 1, 37))

    def fake_init(self, *args, **kwargs):
        self.device = "cpu"
        self.input_height = 32
        self.input_width = 128
        self.beam_width = 1
        self.vocab = {0: 0, "A": 1}
        self.model = FakeModel()

    monkeypatch.setattr(PlateReader, "__init__", fake_init)
    reader = PlateReader("unused")

    reader.read_plate(np.full((24, 56, 3), 255, dtype=np.uint8))

    assert captured_shape == (1, 1, 32, 128)
