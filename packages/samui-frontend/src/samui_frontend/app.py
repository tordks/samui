"""Streamlit application entry point."""

import streamlit as st

from samui_frontend.pages import annotation, processing, upload


st.set_page_config(
    page_title="SAM3 WebUI",
    page_icon="🎯",
    layout="wide",
)

st.title("SAM3 WebUI")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Upload", "Annotation", "Processing"],
    index=0,
)

# Render selected page
if page == "Upload":
    upload.render()
elif page == "Annotation":
    annotation.render()
elif page == "Processing":
    processing.render()
