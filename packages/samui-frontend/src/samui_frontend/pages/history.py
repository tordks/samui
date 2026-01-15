"""Image History page showing processing results with mask+bbox overlays."""

import io
from datetime import datetime

import streamlit as st
from PIL import Image, ImageDraw

from samui_frontend.api import fetch_all_history, fetch_result_mask
from samui_frontend.components import (
    GalleryConfig,
    image_gallery,
    render_mode_toggle,
)


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


def _history_label(result: dict) -> str | None:
    """Generate label for history gallery item showing discoveries and prompt info."""
    parts = []

    bbox_count = len(result.get("bboxes") or [])
    if bbox_count > 0:
        parts.append(f"{bbox_count} discoveries")

    text_prompt = result.get("text_prompt_used")
    if text_prompt:
        prompt_display = f'"{text_prompt[:30]}..."' if len(text_prompt) > 30 else f'"{text_prompt}"'
        parts.append(f"Prompt: {prompt_display}")

    return " | ".join(parts) if parts else None


def _render_history_gallery(results: list[dict]) -> None:
    """Render the history gallery showing processing results with overlays."""
    if not results:
        st.info("No processing history in the current mode.")
        return

    st.subheader(f"Processing Results ({len(results)})")

    # Transform results for gallery:
    # - image_id: already present in result from API
    # - id: result's own id (used for unique widget keys)
    # - filename: formatted timestamp for caption
    gallery_items = [
        {
            **result,
            "filename": _format_timestamp(result.get("processed_at", "")),
        }
        for result in results
    ]

    image_gallery(
        gallery_items,
        config=GalleryConfig(columns=4, show_dimensions=False, key_prefix="history_"),
        image_renderer=_create_history_overlay,
        label_callback=_history_label,
    )


def render() -> None:
    """Render the Image History page."""
    st.header("Processing Results")
    st.caption("View all processing results")

    # Mode toggle at the top
    current_mode = render_mode_toggle(key="history_mode_radio")
    st.divider()

    # Fetch and display all processing history
    history = fetch_all_history(current_mode)
    _render_history_gallery(history)
