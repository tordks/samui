"""Reusable tiled image gallery component."""

from collections.abc import Callable
from typing import Any

import streamlit as st

from samui_frontend.config import API_URL


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
            # Display image thumbnail
            image_url = f"{API_URL}/images/{image['id']}/data"
            st.image(image_url, caption=image["filename"], use_container_width=True)

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
