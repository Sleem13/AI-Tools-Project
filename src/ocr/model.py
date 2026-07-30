"""CRNN (CNN + BiLSTM + CTC) model for license plate OCR."""

from __future__ import annotations

import torch
import torch.nn as nn


class CRNNModel(nn.Module):
    """CRNN for image-to-sequence recognition.

    Architecture:
      - CNN backbone for feature extraction
      - Bidirectional LSTM for sequence modelling
      - CTC linear head for character prediction
    """

    def __init__(
        self,
        num_classes: int,
        input_height: int = 32,
        cnn_channels: list[int] | None = None,
        bilstm_hidden: int = 256,
        bilstm_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [1, 64, 128, 256, 512, 512]

        self.num_classes = num_classes
        self.input_height = input_height

        # CNN backbone
        cnn_layers: list[nn.Module] = []
        in_ch = cnn_channels[0]
        for out_ch in cnn_channels[1:]:
            cnn_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ])
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # Compute flat feature width after CNN
        dummy = torch.zeros(1, cnn_channels[0], input_height, 128)
        with torch.no_grad():
            cnn_out = self.cnn(dummy)
        feat_h, _feat_w = cnn_out.shape[2:]
        cnn_feat_dim = cnn_channels[-1] * feat_h

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=cnn_feat_dim,
            hidden_size=bilstm_hidden,
            num_layers=bilstm_layers,
            bidirectional=True,
            dropout=dropout if bilstm_layers > 1 else 0,
            batch_first=True,
        )

        # CTC head
        self.head = nn.Linear(bilstm_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W)
        features = self.cnn(x)  # (B, C, H', W')
        b, c, h, w = features.shape
        features = features.permute(0, 3, 1, 2).contiguous()  # (B, W', C, H')
        features = features.view(b, w, c * h)  # (B, W', feat_dim)

        lstm_out, _ = self.lstm(features)  # (B, W', 2 * hidden)
        logits = self.head(lstm_out)  # (B, W', num_classes)
        return logits

    def forward_for_ctc(self, x: torch.Tensor) -> torch.Tensor:
        """Return log-softmax output for CTC loss.

        Returns:
            Tensor of shape (T, B, num_classes) as expected by
            torch.nn.CTCLoss.
        """
        logits = self.forward(x)  # (B, T, C)
        log_probs = torch.log_softmax(logits, dim=2)  # (B, T, C)
        return log_probs.permute(1, 0, 2)  # (T, B, C)
