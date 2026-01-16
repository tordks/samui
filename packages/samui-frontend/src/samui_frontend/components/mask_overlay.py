"""Mask overlay component for displaying segmentation results."""

from io import BytesIO

import streamlit as st
from PIL import Image

# Default mask color (semi-transparent green)
MASK_COLOR = (0, 255, 0, 100)


def composite_mask_on_image(
    original: Image.Image,
    mask: Image.Image,
    alpha: int = 50,
    mask_color: tuple[int, int, int] = (0, 255, 0),
) -> Image.Image:
    """Composite a mask onto an image with configurable transparency.

    Args:
        original: Original PIL Image.
        mask: Mask PIL Image (grayscale, white=foreground).
        alpha: Transparency level 0-100 (0=invisible, 100=opaque).
        mask_color: RGB tuple for mask color.

    Returns:
        Image with mask composited.
    """
    # Convert original to RGBA
    img = original.copy().convert("RGBA")

    # Ensure mask is the right size
    if mask.size != img.size:
        mask = mask.resize(img.size, Image.Resampling.NEAREST)

    # Convert mask to grayscale
    mask_gray = mask.convert("L")

    # Create solid color overlay
    overlay = Image.new("RGBA", img.size, (*mask_color, 255))  # type: ignore[arg-type]

    # Blend overlay with original at the specified alpha
    alpha_fraction = alpha / 100
    blended = Image.blend(img, overlay, alpha_fraction)

    # Use mask to select: where mask is white show blended, where black show original
    result = Image.composite(blended, img, mask_gray)

    return result.convert("RGB")


def render_mask_overlay(
    original_bytes: bytes,
    mask_bytes: bytes | None,
    key: str = "mask_overlay",
) -> None:
    """Render an image with optional mask overlay and controls.

    Args:
        original_bytes: Original image as bytes.
        mask_bytes: Mask PNG as bytes, or None if no result.
        key: Unique key prefix for Streamlit widgets.
    """
    # Initialize session state for this component
    show_key = f"{key}_show_overlay"
    alpha_key = f"{key}_alpha"

    if show_key not in st.session_state:
        st.session_state[show_key] = True
    if alpha_key not in st.session_state:
        st.session_state[alpha_key] = 50

    original = Image.open(BytesIO(original_bytes))

    # If no mask, just show original
    if not mask_bytes:
        st.image(original, use_container_width=True)
        st.caption("No processing result available")
        return

    mask = Image.open(BytesIO(mask_bytes)).convert("L")

    # Controls row
    col1, col2 = st.columns([1, 3])

    with col1:
        show_overlay = st.checkbox(
            "Show Mask",
            value=st.session_state[show_key],
            key=f"{key}_checkbox",
        )
        st.session_state[show_key] = show_overlay

    with col2:
        alpha = st.slider(
            "Opacity",
            min_value=0,
            max_value=100,
            value=st.session_state[alpha_key],
            key=f"{key}_slider",
            disabled=not show_overlay,
        )
        st.session_state[alpha_key] = alpha

    # Display image
    result_image = composite_mask_on_image(original, mask, alpha) if show_overlay else original
    st.image(result_image, use_container_width=True)
