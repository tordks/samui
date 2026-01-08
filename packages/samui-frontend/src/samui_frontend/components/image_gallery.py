"""Reusable tiled image gallery component."""

from collections.abc import Callable
from typing import Any

import httpx
import streamlit as st

from samui_frontend.config import API_URL


@st.cache_data(ttl=60)
def _fetch_image_data(image_id: str) -> bytes | None:
    """Fetch image data from the backend API.

    Cached to avoid repeated requests for the same image.
    """
    try:
        response = httpx.get(f"{API_URL}/images/{image_id}/data", timeout=10.0)
        if response.status_code == 200:
            return response.content
    except httpx.RequestError:
        pass
    return None


def image_gallery(
    images: list[dict[str, Any]],
    columns: int = 4,
    on_select: Callable[[dict[str, Any]], None] | None = None,
    show_delete: bool = False,
    on_delete: Callable[[dict[str, Any]], None] | None = None,
) -> str | None:
    """Display a tiled gallery of images.

    Args:
        images: List of image metadata dicts with 'id', 'filename', 'width', 'height'.
        columns: Number of columns in the grid.
        on_select: Callback when an image is selected (receives image dict).
        show_delete: Whether to show delete buttons.
        on_delete: Callback when delete is clicked (receives image dict).

    Returns:
        Selected image ID if any, else None.
    """
    if not images:
        st.info("No images to display")
        return None

    selected_id = None
    cols = st.columns(columns)

    for idx, image in enumerate(images):
        col = cols[idx % columns]
        with col:
            # Fetch and display image (server-side to work with Docker internal URLs)
            image_data = _fetch_image_data(image["id"])
            if image_data:
                st.image(image_data, caption=image["filename"], use_container_width=True)
            else:
                st.warning(f"Failed to load: {image['filename']}")

            # Show dimensions
            st.caption(f"{image['width']}x{image['height']}")

            # Action buttons
            btn_cols = st.columns(2 if show_delete else 1)

            with btn_cols[0]:
                if st.button("Select", key=f"select_{image['id']}"):
                    selected_id = image["id"]
                    if on_select:
                        on_select(image)

            if show_delete and len(btn_cols) > 1:
                with btn_cols[1]:
                    if st.button("Delete", key=f"delete_{image['id']}"):
                        if on_delete:
                            on_delete(image)

    return selected_id
