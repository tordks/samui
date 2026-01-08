"""Bounding box annotator component using streamlit-image-coordinates."""

from collections.abc import Callable
from typing import Any

from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

# Color palette for bounding boxes (cycle through these)
BBOX_COLORS = [
    "#ff4b4b",  # red
    "#4bff4b",  # green
    "#4b4bff",  # blue
    "#ffff4b",  # yellow
    "#ff4bff",  # magenta
    "#4bffff",  # cyan
]


def _draw_bboxes_on_image(
    image: Image.Image,
    annotations: list[dict[str, Any]],
) -> Image.Image:
    """Draw existing bounding boxes on an image.

    Args:
        image: PIL Image to draw on.
        annotations: List of annotation dicts with bbox_x, bbox_y, bbox_width, bbox_height.

    Returns:
        Image with bounding boxes drawn.
    """
    # Make a copy to avoid modifying the original
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    for idx, annotation in enumerate(annotations):
        color = BBOX_COLORS[idx % len(BBOX_COLORS)]
        x1 = annotation["bbox_x"]
        y1 = annotation["bbox_y"]
        x2 = x1 + annotation["bbox_width"]
        y2 = y1 + annotation["bbox_height"]

        # Draw rectangle with thicker border for visibility
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label background
        label = f"Box {idx + 1}"
        label_bbox = draw.textbbox((x1, y1 - 20), label)
        draw.rectangle(label_bbox, fill=color)
        draw.text((x1, y1 - 20), label, fill="white")

    return img_copy


def bbox_annotator(
    image: Image.Image,
    annotations: list[dict[str, Any]],
    on_bbox_drawn: Callable[[int, int, int, int], None] | None = None,
    key: str = "bbox_annotator",
) -> dict[str, int] | None:
    """Interactive bounding box annotator component.

    Displays an image with existing bounding boxes overlaid and allows drawing
    new bounding boxes via click-and-drag.

    Args:
        image: PIL Image to annotate.
        annotations: List of existing annotation dicts with bbox_x, bbox_y, bbox_width, bbox_height.
        on_bbox_drawn: Callback when a new bbox is drawn, receives (x, y, width, height).
        key: Unique key for the Streamlit component.

    Returns:
        Dict with x, y, width, height of newly drawn bbox, or None if no new bbox.
    """
    # Draw existing bboxes on the image
    annotated_image = _draw_bboxes_on_image(image, annotations)

    # Display the image with click-and-drag enabled
    value = streamlit_image_coordinates(
        annotated_image,
        key=key,
        click_and_drag=True,
    )

    # Check if a new rectangle was drawn
    if value and "x1" in value and "x2" in value:
        x1 = value["x1"]
        y1 = value["y1"]
        x2 = value["x2"]
        y2 = value["y2"]

        # Ensure we have a valid rectangle (not just a click)
        if x1 != x2 and y1 != y2:
            # Normalize coordinates (ensure x1 < x2 and y1 < y2)
            min_x = min(x1, x2)
            max_x = max(x1, x2)
            min_y = min(y1, y2)
            max_y = max(y1, y2)

            width = max_x - min_x
            height = max_y - min_y

            bbox = {
                "x": min_x,
                "y": min_y,
                "width": width,
                "height": height,
            }

            if on_bbox_drawn:
                on_bbox_drawn(min_x, min_y, width, height)

            return bbox

    return None


def get_bbox_color(index: int) -> str:
    """Get the color for a bounding box by its index."""
    return BBOX_COLORS[index % len(BBOX_COLORS)]
