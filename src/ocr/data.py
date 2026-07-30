"""OCR dataset — cropped plate images mapped to transcribed text."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Characters that appear on Egyptian license plates (Arabic numerals + Latin letters).
DEFAULT_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
CTC_BLANK = 0


def build_vocab(chars: str | None = None) -> dict[str, int]:
    """Build character-to-index mapping for CTC loss.

    Index 0 is reserved for the CTC blank token.
    """
    if chars is None:
        chars = DEFAULT_CHARS
    vocab = {CTC_BLANK: 0}
    for i, c in enumerate(chars, start=1):
        vocab[c] = i
    return vocab


def encode_text(text: str, vocab: dict[str, int]) -> list[int]:
    """Encode a string into a list of vocab indices."""
    return [vocab[c] for c in text if c in vocab]


def decode_text(indices: list[int], vocab: dict[str, int]) -> str:
    """Decode a list of vocab indices back to a string (inverse of encode_text)."""
    rev = {v: k for k, v in vocab.items()}
    return "".join(rev.get(i, "") for i in indices if i != CTC_BLANK)


class OCRPlateDataset(Dataset):
    """Dataset of cropped license plate images with ground-truth text.

    Expects a JSON annotation file with entries:
        {"image_path": "...", "text": "ABC123"}
    """

    def __init__(
        self,
        annotation_path: str | Path,
        image_root: str | Path | None = None,
        vocab: dict[str, int] | None = None,
        input_height: int = 32,
        input_width: int = 128,
        augment: bool = False,
    ) -> None:
        self.image_root = Path(image_root) if image_root else None
        self.input_height = input_height
        self.input_width = input_width
        self.augment = augment
        self.vocab = vocab or build_vocab()

        with open(annotation_path, encoding="utf-8") as f:
            self.samples: list[dict] = json.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        img_path = Path(sample["image_path"])
        if self.image_root:
            img_path = self.image_root / img_path

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot read OCR crop: {img_path}")

        image = self._preprocess(image)
        text = sample.get("text", "")
        target = encode_text(text.upper(), self.vocab)

        image_tensor = torch.from_numpy(image).unsqueeze(0).float() / 255.0
        target_tensor = torch.tensor(target, dtype=torch.long)

        return image_tensor, target_tensor

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        target_h = self.input_height
        target_w = self.input_width

        scale = target_h / h
        new_w = round(w * scale)
        if new_w == 0:
            new_w = 1
        resized = cv2.resize(image, (new_w, target_h), interpolation=cv2.INTER_AREA)

        if new_w < target_w:
            pad_w = target_w - new_w
            resized = cv2.copyMakeBorder(resized, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=255)
        else:
            resized = resized[:, :target_w]

        return resized
