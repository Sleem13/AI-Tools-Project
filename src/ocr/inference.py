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
        input_width: int = 128,
        beam_width: int = 1,
    ) -> None:
        self.device = torch.device(device)
        self.input_height = input_height
        self.input_width = input_width
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
        new_w = round(w * scale)
        if new_w == 0:
            new_w = 1
        resized = cv2.resize(gray, (new_w, self.input_height), interpolation=cv2.INTER_AREA)
        if new_w < self.input_width:
            resized = cv2.copyMakeBorder(
                resized,
                0,
                0,
                0,
                self.input_width - new_w,
                cv2.BORDER_CONSTANT,
                value=255,
            )
        else:
            resized = resized[:, : self.input_width]

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
        """Beam search decoder for CTC output.

        Maintains top-k partial sequences at each timestep, expanding with
        the highest-probability next characters.  Collapses consecutive
        duplicate characters and strips blanks per CTC convention.
        """
        time_steps, class_count = probs.shape
        blank = 0

        # Each beam entry: (cumulative_log_prob, previous_index, current_sequence)
        # Start with a single beam containing only the blank token.
        beams: list[tuple[float, int, list[int]]] = [(0.0, blank, [])]

        for t in range(time_steps):
            candidates: dict[tuple[int, ...], tuple[float, int]] = {}
            for score, _prev_idx, seq in beams:
                for c in range(class_count):
                    log_p = float(probs[t, c].log().item())
                    new_score = score + log_p

                    if c == blank:
                        key = tuple(seq)
                    elif seq and seq[-1] == c:
                        # CTC: extending the same character collapses, so we
                        # keep the same sequence but the probability accumulates.
                        key = tuple(seq)
                    else:
                        key = tuple([*seq, c])

                    if key not in candidates or new_score > candidates[key][0]:
                        candidates[key] = (new_score, c)

            # Keep only the top beam_width candidates
            sorted_cands = sorted(candidates.items(), key=lambda x: x[1][0], reverse=True)
            beams = [
                (score, prev_idx, list(seq))
                for seq, (score, prev_idx) in sorted_cands[: self.beam_width]
            ]

        # Pick the best beam
        best_seq = max(beams, key=lambda x: x[0])[2]
        return decode_text(best_seq, self.vocab)
