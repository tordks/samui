"""Shared utility functions."""

from samui_backend.db.models import SegmentationMode


def get_image_content_type(extension: str) -> str:
    """Get MIME content type for an image extension.

    Args:
        extension: File extension (with or without leading dot).

    Returns:
        MIME content type string.
    """
    ext = extension.lower().lstrip(".")
    if ext == "jpg":
        return "image/jpeg"
    return f"image/{ext}"


def get_blob_path_suffix(mode: SegmentationMode) -> str:
    """Get filename suffix for non-default segmentation modes.

    Args:
        mode: The segmentation mode.

    Returns:
        Empty string for INSIDE_BOX mode, otherwise "_{mode.value}".
    """
    if mode == SegmentationMode.INSIDE_BOX:
        return ""
    return f"_{mode.value}"
