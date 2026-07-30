"""Keras CRNN inference for notebook-trained Egyptian plate OCR weights."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import cv2
import numpy as np

IMG_W = 256
IMG_H = 128
MAX_TEXT_LEN = 7

NOTEBOOK_LABEL_MAP = {
    "0": "0",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "alif": "A",
    "baa": "B",
    "taa": "C",
    "thaa": "D",
    "jeem": "E",
    "7aa": "F",
    "khaa": "G",
    "daal": "H",
    "zaal": "I",
    "raa": "J",
    "zay": "K",
    "seen": "L",
    "sheen": "M",
    "saad": "N",
    "daad": "O",
    "taa_cap": "P",
    "tah": "P",
    "zaa": "Q",
    "ain": "R",
    "ghayn": "S",
    "faa": "T",
    "qaaf": "U",
    "kaaf": "V",
    "laam": "W",
    "meem": "n",
    "noon": "Y",
    "haa": "Z",
    "waw": "a",
    "yaa": "b",
    "ا": "A",
    "ب": "B",
    "ت": "C",
    "ث": "D",
    "ج": "E",
    "ح": "F",
    "خ": "G",
    "د": "H",
    "ذ": "I",
    "ر": "J",
    "ز": "K",
    "س": "L",
    "ش": "M",
    "ص": "N",
    "ض": "O",
    "ط": "P",
    "ظ": "Q",
    "ع": "R",
    "غ": "S",
    "ف": "T",
    "ق": "U",
    "ك": "V",
    "ل": "W",
    "م": "n",
    "ن": "Y",
    "ه": "Z",
    "و": "a",
    "ى": "b",
}

UNIQUE_LETTERS = [*sorted(set(NOTEBOOK_LABEL_MAP.values())), "X"]
CHAR_VECTOR = "".join(UNIQUE_LETTERS)
NUM_CLASSES = len(UNIQUE_LETTERS) + 1

EN_TO_DISPLAY = {
    "A": "ا",
    "B": "ب",
    "C": "ت",
    "D": "ث",
    "E": "ج",
    "F": "ح",
    "G": "خ",
    "H": "د",
    "I": "ذ",
    "J": "ر",
    "K": "ز",
    "L": "س",
    "M": "ش",
    "N": "ص",
    "O": "ض",
    "P": "ط",
    "Q": "ظ",
    "R": "ع",
    "S": "غ",
    "T": "ف",
    "U": "ق",
    "V": "ك",
    "W": "ل",
    "n": "م",
    "Y": "ن",
    "Z": "ه",
    "a": "و",
    "b": "ي",
    "0": "0",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
}


def _load_tensorflow():
    try:
        import tensorflow as tf
        from tensorflow.keras import Model, layers
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorFlow is required to load notebook OCR weights. "
            "Install it with `pip install -r requirements-keras-ocr.txt`."
        ) from exc
    return tf, Model, layers


def build_keras_crnn_model() -> Any:
    """Build the inference graph from egyptian-car-plates-ocr-crnn.ipynb."""
    _tf, model_cls, layers = _load_tensorflow()

    inputs = layers.Input(name="the_input", shape=(IMG_W, IMG_H, 1))

    x = layers.GaussianNoise(stddev=0.03)(inputs)
    x = layers.RandomRotation(factor=0.04, fill_mode="nearest")(x)
    x = layers.RandomTranslation(height_factor=0.04, width_factor=0.06)(x)
    x = layers.RandomZoom(height_factor=(-0.08, 0.08), width_factor=(-0.08, 0.08))(x)
    x = layers.RandomContrast(factor=0.2)(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.15)(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.15)(x)

    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)

    conv_shape = x.shape
    x = layers.Reshape(
        target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])),
        name="reshape",
    )(x)

    x = layers.Dense(512, activation="relu", name="dense_bridge")(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.1))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.1))(x)
    x = layers.Dense(NUM_CLASSES, kernel_initializer="he_normal")(x)
    y_pred = layers.Activation("softmax", name="softmax")(x)

    return model_cls(inputs=[inputs], outputs=y_pred)


class KerasPlateReader:
    """Read plate text using the TensorFlow/Keras CRNN exported by the notebook."""

    source = "keras_crnn"

    def __init__(self, weights_path: str | Path) -> None:
        self.weights_path = Path(weights_path)
        self.model = build_keras_crnn_model()
        self.model.load_weights(str(self.weights_path))

    def read_plate(self, image: np.ndarray) -> str:
        """Recognise text from a cropped plate image."""
        if image is None:
            return ""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(gray, -1, kernel)
        resized = cv2.resize(sharpened, (IMG_W, IMG_H))
        model_input = resized.T.reshape(1, IMG_W, IMG_H, 1).astype(np.float32) / 255.0

        preds = self.model.predict(model_input, verbose=0)
        return self.decode_prediction(preds[0])

    @staticmethod
    def decode_prediction(prediction: np.ndarray) -> str:
        best_path = np.argmax(prediction, axis=1)
        deduped = [idx for idx, _group in itertools.groupby(best_path)]
        raw_text = "".join(
            CHAR_VECTOR[idx]
            for idx in deduped
            if idx < len(UNIQUE_LETTERS) and CHAR_VECTOR[idx] != "X"
        )
        return KerasPlateReader.to_display_text(raw_text[::-1])

    @staticmethod
    def to_display_text(encoded_text: str) -> str:
        chars = [EN_TO_DISPLAY[char] for char in encoded_text if char in EN_TO_DISPLAY]
        return " ".join(chars).strip()
