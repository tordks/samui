"""Shared utility functions."""


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
