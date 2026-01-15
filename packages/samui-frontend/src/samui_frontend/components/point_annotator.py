"""Point annotator component using streamlit-image-coordinates."""

from typing import Any

import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

from samui_frontend.constants import COLOR_NEGATIVE_EXEMPLAR, COLOR_POSITIVE_EXEMPLAR

# Point rendering constants
POINT_RADIUS = 6
POINT_OUTLINE_WIDTH = 2
POINT_HIT_THRESHOLD = 15  # pixels for hit detection in delete mode


def _draw_points_on_image(
    image: Image.Image,
    points: list[dict[str, Any]],
) -> Image.Image:
    """Draw existing points on an image.

    Args:
        image: PIL Image to draw on.
        points: List of point annotation dicts with point_x, point_y, is_positive.

    Returns:
        Image with points drawn as colored circles.
    """
    img_copy = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(img_copy)

    for point in points:
        x = point["point_x"]
        y = point["point_y"]
        is_positive = point["is_positive"]

        fill_color = COLOR_POSITIVE_EXEMPLAR if is_positive else COLOR_NEGATIVE_EXEMPLAR

        # Draw filled circle with white outline
        draw.ellipse(
            [x - POINT_RADIUS, y - POINT_RADIUS, x + POINT_RADIUS, y + POINT_RADIUS],
            fill=fill_color,
            outline="white",
            width=POINT_OUTLINE_WIDTH,
        )

    return img_copy.convert("RGB")


def _find_nearest_point(
    click_x: int,
    click_y: int,
    points: list[dict[str, Any]],
    threshold: int = POINT_HIT_THRESHOLD,
) -> dict[str, Any] | None:
    """Find the nearest point within threshold distance of a click.

    Args:
        click_x: X coordinate of the click.
        click_y: Y coordinate of the click.
        points: List of point annotation dicts.
        threshold: Maximum distance in pixels to consider a hit.

    Returns:
        The nearest point dict if within threshold, None otherwise.
    """
    nearest_point = None
    min_distance = float("inf")

    for point in points:
        px = point["point_x"]
        py = point["point_y"]
        distance = ((click_x - px) ** 2 + (click_y - py) ** 2) ** 0.5

        if distance < min_distance and distance <= threshold:
            min_distance = distance
            nearest_point = point

    return nearest_point


def point_annotator(
    image: Image.Image,
    points: list[dict[str, Any]],
    key: str = "point_annotator",
) -> dict[str, int] | None:
    """Interactive point annotator component.

    Displays an image with existing points overlaid and detects clicks
    to add new points.

    Args:
        image: PIL Image to annotate.
        points: List of existing point annotation dicts with point_x, point_y, is_positive.
        key: Unique key for the Streamlit component.

    Returns:
        Dict with x, y coordinates of the click, or None if no new click.
    """
    # Track last processed click timestamp to avoid processing the same click twice
    state_key = f"_last_click_time_{key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = None

    # Draw existing points on the image
    annotated_image = _draw_points_on_image(image, points)

    # Display the image with click detection (no drag needed for points)
    value = streamlit_image_coordinates(
        annotated_image,
        key=key,
        click_and_drag=False,
    )

    # Check if a click occurred
    if value and "x" in value and "y" in value:
        current_time = value.get("unix_time")

        # Skip if we've already processed this exact click (same timestamp)
        if current_time is not None and current_time == st.session_state[state_key]:
            return None

        # Mark this click as processed
        st.session_state[state_key] = current_time

        return {
            "x": value["x"],
            "y": value["y"],
        }

    return None


def find_point_at_click(
    click_x: int,
    click_y: int,
    points: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Find a point near the click coordinates for deletion.

    Args:
        click_x: X coordinate of the click.
        click_y: Y coordinate of the click.
        points: List of point annotation dicts.

    Returns:
        The nearest point dict if within hit threshold, None otherwise.
    """
    return _find_nearest_point(click_x, click_y, points)
