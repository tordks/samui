"""Streamlit application entry point."""

import streamlit as st

from samui_frontend.models import PromptType, SegmentationMode
from samui_frontend.pages import annotation, history, processing, upload


def _init_session_state() -> None:
    """Initialize all session state variables with defaults."""
    defaults = {
        "selected_image_index": 0,
        "segmentation_mode": SegmentationMode.INSIDE_BOX,
        "exemplar_type": PromptType.POSITIVE_EXEMPLAR,
        "selected_processed_index": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


st.set_page_config(
    page_title="SAM3 WebUI",
    page_icon="🎯",
    layout="wide",
)

st.title("SAM3 WebUI")

# Initialize session state
_init_session_state()

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Upload", "Annotation", "Processing", "History"],
    index=0,
)

# Render selected page
if page == "Upload":
    upload.render()
elif page == "Annotation":
    annotation.render()
elif page == "Processing":
    processing.render()
elif page == "History":
    history.render()
