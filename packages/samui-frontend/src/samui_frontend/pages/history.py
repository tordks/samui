"""Image History page showing processing results with mask+bbox overlays."""

import io
from datetime import datetime

import streamlit as st
from PIL import Image, ImageDraw

from samui_frontend.api import (
    fetch_image_data,
    fetch_image_history,
    fetch_images,
    fetch_result_mask,
)
from samui_frontend.components import arrow_navigator, render_mode_toggle


def _create_history_overlay(result: dict, image_data: bytes) -> Image.Image:
    """Create an image overlay with mask and discovered bboxes from a processing result.

    Args:
        result: Processing result dict with bboxes and mask info.
        image_data: Original image bytes.

    Returns:
        PIL Image with mask and bbox overlays applied.
    """
    # Load the original image
    image = Image.open(io.BytesIO(image_data)).convert("RGBA")

    # Fetch and overlay mask
    mask_data = fetch_result_mask(result["id"])
    if mask_data:
        mask = Image.open(io.BytesIO(mask_data)).convert("L")

        # Resize mask to match image if needed
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.NEAREST)

        # Where mask is white (255), overlay semi-transparent green
        mask_rgba = Image.new("RGBA", image.size, (0, 255, 0, 100))
        image = Image.composite(mask_rgba, image, mask)

    # Draw discovered bboxes from result
    bboxes = result.get("bboxes") or []
    if bboxes:
        draw = ImageDraw.Draw(image)
        for idx, bbox in enumerate(bboxes):
            # bbox format: {x, y, width, height}
            x = bbox.get("x", 0)
            y = bbox.get("y", 0)
            width = bbox.get("width", 0)
            height = bbox.get("height", 0)

            # Draw rectangle in cyan for discovered boxes
            color = (0, 255, 255)  # Cyan for model discoveries
            draw.rectangle([x, y, x + width, y + height], outline=color, width=2)

            # Draw label
            label = f"D{idx + 1}"
            label_bbox = draw.textbbox((x, y - 16), label)
            draw.rectangle(label_bbox, fill=color)
            draw.text((x, y - 16), label, fill="black")

    return image.convert("RGB")


def _format_timestamp(timestamp_str: str) -> str:
    """Format ISO timestamp to human-readable string."""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError):
        return timestamp_str or "Unknown"


def _render_history_gallery(
    results: list[dict],
    image_data: bytes,
) -> None:
    """Render the history gallery showing processing results with overlays."""
    if not results:
        st.info("No processing history for this image in the current mode.")
        return

    st.subheader(f"Processing History ({len(results)} results)")

    # Create columns for gallery display
    cols_per_row = min(len(results), 4)
    cols = st.columns(cols_per_row)

    for idx, result in enumerate(results):
        col = cols[idx % cols_per_row]
        with col:
            # Create overlay image
            overlay = _create_history_overlay(result, image_data)
            timestamp = _format_timestamp(result.get("processed_at", ""))

            # Show the overlaid image with timestamp as caption
            st.image(overlay, caption=timestamp, use_container_width=True)

            # Show additional info
            bbox_count = len(result.get("bboxes") or [])
            if bbox_count > 0:
                st.caption(f"{bbox_count} discoveries")

            text_prompt = result.get("text_prompt_used")
            if text_prompt:
                st.caption(f'Prompt: "{text_prompt[:30]}..."' if len(text_prompt) > 30 else f'Prompt: "{text_prompt}"')


def render() -> None:
    """Render the Image History page."""
    st.header("Image History")
    st.caption("View processing results and history for each image")

    # Mode toggle at the top
    current_mode = render_mode_toggle(key="history_mode_radio")
    st.divider()

    # Fetch all images
    images = fetch_images()

    if not images:
        st.info("No images uploaded yet. Upload images first.")
        return

    # Arrow navigator for image selection
    st.subheader("Select Image")
    current_index = arrow_navigator(images, "history_image")

    if current_index is None:
        return

    selected_image = images[current_index]
    st.write(f"**{selected_image['filename']}** ({selected_image['width']}x{selected_image['height']})")

    st.divider()

    # Fetch and display original image
    image_data = fetch_image_data(selected_image["id"])
    if not image_data:
        st.error("Failed to load image data.")
        return

    # Two-column layout: original image on left, info on right
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Original Image")
        st.image(image_data, use_container_width=True)

    with col2:
        st.subheader("Image Info")
        st.write(f"**Filename:** {selected_image['filename']}")
        st.write(f"**Dimensions:** {selected_image['width']} x {selected_image['height']}")
        if selected_image.get("text_prompt"):
            st.write(f"**Text Prompt:** {selected_image['text_prompt']}")

    st.divider()

    # Fetch and display processing history
    history = fetch_image_history(selected_image["id"], current_mode)
    _render_history_gallery(history, image_data)
