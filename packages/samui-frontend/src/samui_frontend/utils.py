"""Shared utility functions for the frontend."""

from PIL import Image

from samui_frontend.constants import (
    BBOX_COLORS,
    COLOR_NEGATIVE_EXEMPLAR,
    COLOR_POSITIVE_EXEMPLAR,
)
from samui_frontend.models import PromptType


def get_annotation_color(annotation: dict, index: int) -> str:
    """Get color for annotation based on prompt_type.

    Args:
        annotation: Annotation dict with prompt_type field.
        index: Index for cycling through default colors.

    Returns:
        Hex color string.
    """
    prompt_type = annotation.get("prompt_type", PromptType.SEGMENT.value)
    if prompt_type == PromptType.POSITIVE_EXEMPLAR.value:
        return COLOR_POSITIVE_EXEMPLAR
    elif prompt_type == PromptType.NEGATIVE_EXEMPLAR.value:
        return COLOR_NEGATIVE_EXEMPLAR
    return BBOX_COLORS[index % len(BBOX_COLORS)]


def get_text_prompt_label(image: dict) -> str | None:
    """Return text prompt label for gallery display.

    Args:
        image: Image dict with optional text_prompt field.

    Returns:
        Label string or None if no text prompt.
    """
    text_prompt = image.get("text_prompt")
    if text_prompt:
        return f"text prompt: {text_prompt}"
    return None


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
