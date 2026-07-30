"""OCR module for license plate text recognition.

Provides:
  - OCRPlateDataset: cropped plate image + text sequence dataset
  - CRNNModel: CNN + BiLSTM + CTC architecture
  - PlateReader: inference wrapper for trained OCR models
  - train_crnn: programmatic CRNN training entry point
"""

from src.ocr.data import OCRPlateDataset, build_vocab, decode_text, encode_text
from src.ocr.inference import PlateReader
from src.ocr.keras_inference import KerasPlateReader
from src.ocr.model import CRNNModel
from src.ocr.trainer import train_crnn

__all__ = [
    "CRNNModel",
    "KerasPlateReader",
    "OCRPlateDataset",
    "PlateReader",
    "build_vocab",
    "decode_text",
    "encode_text",
    "train_crnn",
]
