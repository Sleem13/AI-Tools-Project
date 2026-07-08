"""OCR inference — load a trained CRNN and read plate text from crops."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from src.ocr.data import build_vocab, decode_text
from src.ocr.model import CRNNModel


class PlateReader:
    """Load a trained CRNN model and decode license plate text from images."""

    def __init__(
        self,
        weights_path: str | Path,
        device: str = "cpu",
        input_height: int = 32,
        beam_width: int = 1,
    ) -> None:
        self.device = torch.device(device)
        self.input_height = input_height
        self.beam_width = beam_width
        self.vocab = build_vocab()

        self.model = CRNNModel(
            num_classes=len(self.vocab),
            input_height=input_height,
        ).to(self.device)
        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    def read_plate(self, image: np.ndarray) -> str:
        """Recognise text from a cropped plate image.

        Args:
            image: BGR or grayscale plate crop.

        Returns:
            Decoded plate text.
        """
        if image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        h, w = gray.shape
        scale = self.input_height / h
        new_w = int(round(w * scale))
        if new_w == 0:
            new_w = 1
        resized = cv2.resize(gray, (new_w, self.input_height), interpolation=cv2.INTER_AREA)

        tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)

        with torch.no_grad():
            log_probs = self.model.forward_for_ctc(tensor)  # (T, 1, C)
            probs = log_probs.exp().squeeze(1)  # (T, C)

        if self.beam_width > 1:
            return self._beam_search(probs)
        return self._greedy_decode(probs)

    def _greedy_decode(self, probs: torch.Tensor) -> str:
        indices = probs.argmax(dim=1).tolist()
        collapsed: list[int] = []
        prev = None
        for idx in indices:
            if idx != prev and idx != 0:
                collapsed.append(idx)
            prev = idx
        return decode_text(collapsed, self.vocab)

    def _beam_search(self, probs: torch.Tensor) -> str:
        """Simple beam search decoder (placeholder)."""
        return self._greedy_decode(probs)
