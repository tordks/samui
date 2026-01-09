"""Upload page for image uploads and gallery display."""

import streamlit as st

from samui_frontend.api import delete_image, fetch_images, upload_image
from samui_frontend.components.image_gallery import GalleryConfig, image_gallery


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

            result = upload_image(
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "application/octet-stream",
            )
            if not result:
                st.error(f"Failed to upload {uploaded_file.name}")
                continue

            progress_bar.progress((idx + 1) / len(uploaded_files))

        status_text.text("Upload complete!")
        st.success(f"Successfully uploaded {len(uploaded_files)} image(s)")
        st.rerun()

    st.divider()

    # Image gallery
    st.subheader("Uploaded Images")
    images = fetch_images()

    def handle_delete(image: dict) -> None:
        """Handle image deletion."""
        if delete_image(image["id"]):
            st.success(f"Deleted {image['filename']}")
            st.rerun()
        else:
            st.error("Failed to delete image")

    image_gallery(
        images,
        config=GalleryConfig(columns=4, show_delete=True),
        on_delete=handle_delete,
    )
