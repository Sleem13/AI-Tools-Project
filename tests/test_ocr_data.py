"""Unit tests for OCR data utilities (vocab, encode/decode)."""

from __future__ import annotations

import numpy as np
import pytest


class TestBuildVocab:
    def test_blank_token_at_index_zero(self) -> None:
        from src.ocr.data import CTC_BLANK, build_vocab

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
        from src.ocr.data import build_vocab, decode_text, encode_text

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
        from src.ocr.data import CTC_BLANK, build_vocab, decode_text

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


def test_keras_notebook_decoder_matches_custom_alphabet() -> None:
    from src.ocr.keras_inference import CHAR_VECTOR, NUM_CLASSES, UNIQUE_LETTERS, KerasPlateReader
    from src.postprocessing.plate_formatter import format_plate

    blank_idx = len(UNIQUE_LETTERS)
    encoded_path = [
        CHAR_VECTOR.index("5"),
        CHAR_VECTOR.index("5"),
        blank_idx,
        CHAR_VECTOR.index("3"),
        CHAR_VECTOR.index("X"),
        CHAR_VECTOR.index("A"),
        CHAR_VECTOR.index("Y"),
        CHAR_VECTOR.index("E"),
    ]
    prediction = np.zeros((len(encoded_path), NUM_CLASSES), dtype=np.float32)
    for row, class_idx in enumerate(encoded_path):
        prediction[row, class_idx] = 1.0

    text = KerasPlateReader.decode_prediction(prediction)

    assert text == "ج ن ا 3 5"
    assert format_plate(text) == "جنا 35"
