"""OCR module for license plate text recognition.

Provides:
  - OCRPlateDataset: cropped plate image + text sequence dataset
  - CRNNModel: CNN + BiLSTM + CTC architecture
  - PlateReader: inference wrapper for trained OCR models
  - train_crnn: programmatic CRNN training entry point
"""

from src.ocr.data import OCRPlateDataset, build_vocab, encode_text, decode_text
from src.ocr.model import CRNNModel
from src.ocr.inference import PlateReader
from src.ocr.trainer import train_crnn

__all__ = [
    "OCRPlateDataset",
    "build_vocab",
    "encode_text",
    "decode_text",
    "CRNNModel",
    "PlateReader",
    "train_crnn",
]
