"""ALPR Dashboard — Upload image, see detection + OCR results.

Usage:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.inference import LicensePlateDetector
from src.evaluation.pipeline import ALPRPipeline
from src.ocr.inference import PlateReader
from src.postprocessing.plate_formatter import format_plate
from src.visualization.detection_viz import create_dashboard_figure

DET_WEIGHTS = PROJECT_ROOT / "models" / "weights" / "detection_best.pt"
OCR_WEIGHTS = PROJECT_ROOT / "models" / "weights" / "ocr_best.pth"


@st.cache_resource
def load_pipeline():
    detector = LicensePlateDetector(
        weights_path=DET_WEIGHTS,
        conf_threshold=0.25,
    )
    reader = PlateReader(
        weights_path=OCR_WEIGHTS,
    )
    return ALPRPipeline(detector=detector, reader=reader)


st.set_page_config(page_title="ALPR Dashboard", layout="wide")
st.title("ALPR — License Plate Detection & Recognition")

with st.sidebar:
    st.header("Settings")
    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)
    st.markdown("---")
    st.markdown("### Model Status")
    det_ok = DET_WEIGHTS.exists()
    ocr_ok = OCR_WEIGHTS.exists()
    st.write(f"Detection: {'✅' if det_ok else '❌'} {DET_WEIGHTS.name if det_ok else 'not found'}")
    st.write(f"OCR: {'✅' if ocr_ok else '❌'} {OCR_WEIGHTS.name if ocr_ok else 'not found'}")

st.info("Upload a vehicle image to detect license plates and read the text.")

uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        st.error("Could not decode image.")
        st.stop()

    if not DET_WEIGHTS.exists():
        st.warning("Detection model weights not found. Displaying placeholder.")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        st.stop()

    pipeline = load_pipeline()
    results = pipeline.run(image)

    if results:
        st.success(f"Detected {len(results)} plate(s)")
        for r in results:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.metric("Raw Text", r["plate_text"])
            with col2:
                st.metric("Formatted", format_plate(r["plate_text"]))
            st.caption(f"Confidence: {r['confidence']:.2%}")

        annotated = create_dashboard_figure(image, results)
        st.image(annotated, caption="Detection Results", use_container_width=True)
    else:
        st.warning("No license plates detected.")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
