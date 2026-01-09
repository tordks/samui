"""Upload page for image uploads and gallery display."""

import httpx
import streamlit as st

from samui_frontend.components.image_gallery import GalleryConfig, image_gallery
from samui_frontend.config import API_URL


def render() -> None:
    """Render the upload page."""
    st.header("Upload Images")

    # File uploader with drag-and-drop
    uploaded_files = st.file_uploader(
        "Drag and drop images or click to browse",
        type=["jpg", "jpeg", "png", "gif", "bmp", "webp"],
        accept_multiple_files=True,
    )

    # Upload button
    if uploaded_files and st.button("Upload Selected Images", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Uploading {uploaded_file.name}...")

            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = httpx.post(f"{API_URL}/images", files=files, timeout=30.0)
                response.raise_for_status()
            except httpx.HTTPError as e:
                st.error(f"Failed to upload {uploaded_file.name}: {e}")
                continue

            progress_bar.progress((idx + 1) / len(uploaded_files))

        status_text.text("Upload complete!")
        st.success(f"Successfully uploaded {len(uploaded_files)} image(s)")
        st.rerun()

    st.divider()

    # Image gallery
    st.subheader("Uploaded Images")

    # Fetch images from API
    try:
        response = httpx.get(f"{API_URL}/images", timeout=10.0)
        response.raise_for_status()
        data = response.json()
        images = data.get("images", [])
    except httpx.HTTPError as e:
        st.error(f"Failed to fetch images: {e}")
        images = []

    def handle_delete(image: dict) -> None:
        """Handle image deletion."""
        try:
            response = httpx.delete(f"{API_URL}/images/{image['id']}", timeout=10.0)
            response.raise_for_status()
            st.success(f"Deleted {image['filename']}")
            st.rerun()
        except httpx.HTTPError as e:
            st.error(f"Failed to delete image: {e}")

    image_gallery(
        images,
        config=GalleryConfig(columns=4, show_delete=True),
        on_delete=handle_delete,
    )
