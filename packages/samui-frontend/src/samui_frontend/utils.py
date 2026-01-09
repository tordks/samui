"""Shared utility functions for the frontend."""

from samui_frontend.constants import (
    BBOX_COLORS,
    COLOR_NEGATIVE_EXEMPLAR,
    COLOR_POSITIVE_EXEMPLAR,
)
from samui_frontend.models import PromptType, SegmentationMode


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


def get_annotation_label(annotation: dict, index: int, mode: SegmentationMode) -> str:
    """Get label for annotation based on prompt_type and mode.

    Args:
        annotation: Annotation dict with prompt_type field.
        index: Index for labeling.
        mode: Current segmentation mode.

    Returns:
        Label string like "Box 1" or "+Ex 1".
    """
    prompt_type = annotation.get("prompt_type", PromptType.SEGMENT.value)
    if mode == SegmentationMode.FIND_ALL:
        if prompt_type == PromptType.POSITIVE_EXEMPLAR.value:
            return f"+Ex {index + 1}"
        elif prompt_type == PromptType.NEGATIVE_EXEMPLAR.value:
            return f"-Ex {index + 1}"
    return f"Box {index + 1}"


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
