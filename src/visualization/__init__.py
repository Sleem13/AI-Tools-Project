"""Visualization utilities for ALPR model outputs.

Provides:
  - draw_detections: overlay bounding boxes + labels on images
  - draw_plate_crops: side-by-side display of detected plate crops
  - create_dashboard_figure: full pipeline visualization for Streamlit
"""

from src.visualization.detection_viz import (
    draw_detections,
    draw_plate_text,
    create_dashboard_figure,
)

__all__ = ["draw_detections", "draw_plate_text", "create_dashboard_figure"]
