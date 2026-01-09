"""Reusable tiled image gallery component."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import streamlit as st
from PIL import Image

from samui_frontend.api import fetch_image_data


@dataclass
class GalleryConfig:
    """Configuration options for image gallery."""

    columns: int = 4
    show_delete: bool = False
    show_dimensions: bool = True
    max_images: int | None = None
    key_prefix: str = ""
    selected_index: int | None = None


@st.cache_data(ttl=60)
def _fetch_image_data_cached(image_id: str) -> bytes | None:
    """Fetch image data with caching for gallery display."""
    return fetch_image_data(image_id)


def _render_image(
    image: dict[str, Any],
    image_data: bytes,
    image_renderer: Callable[[dict[str, Any], bytes], Image.Image] | None,
) -> None:
    """Render an image, optionally with a custom renderer."""
    if image_renderer:
        rendered = image_renderer(image, image_data)
        st.image(rendered, caption=image["filename"], use_container_width=True)
    else:
        st.image(image_data, caption=image["filename"], use_container_width=True)


def _render_action_buttons(
    idx: int,
    image: dict[str, Any],
    config: GalleryConfig,
    on_select: Callable[[dict[str, Any]], None] | None,
    on_delete: Callable[[dict[str, Any]], None] | None,
) -> int | None:
    """Render select and delete buttons. Returns selected index if clicked."""
    selected_idx = None
    btn_cols = st.columns(2 if config.show_delete else 1)

    with btn_cols[0]:
        is_selected = idx == config.selected_index
        button_label = "Selected" if is_selected else "Select"
        if st.button(
            button_label,
            key=f"{config.key_prefix}select_{image['id']}",
            disabled=is_selected,
        ):
            selected_idx = idx
            if on_select:
                on_select(image)

    if config.show_delete and len(btn_cols) > 1:
        with btn_cols[1]:
            if st.button("Delete", key=f"{config.key_prefix}delete_{image['id']}") and on_delete:
                on_delete(image)

    return selected_idx


def image_gallery(
    images: list[dict[str, Any]],
    config: GalleryConfig | None = None,
    on_select: Callable[[dict[str, Any]], None] | None = None,
    on_delete: Callable[[dict[str, Any]], None] | None = None,
    image_renderer: Callable[[dict[str, Any], bytes], Image.Image] | None = None,
    label_callback: Callable[[dict[str, Any]], str | None] | None = None,
) -> int | None:
    """Display a tiled gallery of images.

    Args:
        images: List of image metadata dicts with 'id', 'filename', 'width', 'height'.
        config: Gallery configuration options. Uses defaults if not provided.
        on_select: Callback when an image is selected (receives image dict).
        on_delete: Callback when delete is clicked (receives image dict).
        image_renderer: Optional callback to render custom image (e.g., with overlays).
            Receives (image_meta, raw_bytes) and returns PIL Image.
        label_callback: Optional callback to get a label to display above each image.
            Receives image dict and returns label string or None.

    Returns:
        Selected image index if any, else None.
    """
    if not images:
        st.info("No images to display")
        return None

    config = config or GalleryConfig()
    selected_idx = None
    display_images = images[: config.max_images] if config.max_images else images
    cols = st.columns(min(len(display_images), config.columns))

    for idx, image in enumerate(display_images):
        col = cols[idx % config.columns]
        with col:
            # Show label above image if callback provided
            if label_callback:
                label = label_callback(image)
                if label:
                    st.caption(label)

            image_data = _fetch_image_data_cached(image["id"])
            if image_data:
                _render_image(image, image_data, image_renderer)
            else:
                st.warning(f"Failed to load: {image['filename']}")

            if config.show_dimensions:
                st.caption(f"{image['width']}x{image['height']}")

            result = _render_action_buttons(idx, image, config, on_select, on_delete)
            if result is not None:
                selected_idx = result

    return selected_idx
