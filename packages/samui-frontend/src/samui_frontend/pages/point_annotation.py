"""Point annotation page for placing point prompts on images."""

from io import BytesIO

import streamlit as st
from PIL import Image, ImageDraw

from samui_frontend.api import (
    create_point_annotation,
    delete_point_annotation,
    fetch_image_data,
    fetch_images,
    fetch_point_annotations,
)
from samui_frontend.components.image_gallery import GalleryConfig, image_gallery
from samui_frontend.components.point_annotator import find_point_at_click, point_annotator
from samui_frontend.constants import COLOR_NEGATIVE_EXEMPLAR, COLOR_POSITIVE_EXEMPLAR

# Point rendering for gallery thumbnails
THUMBNAIL_POINT_RADIUS = 4


def _create_point_overlay(image: dict, image_data: bytes) -> Image.Image:
    """Create overlay with points for gallery display."""
    pil_image = Image.open(BytesIO(image_data)).convert("RGBA")
    points = fetch_point_annotations(image["id"])

    if not points:
        return pil_image.convert("RGB")

    draw = ImageDraw.Draw(pil_image)
    for point in points:
        x = point["point_x"]
        y = point["point_y"]
        is_positive = point["is_positive"]
        color = COLOR_POSITIVE_EXEMPLAR if is_positive else COLOR_NEGATIVE_EXEMPLAR

        draw.ellipse(
            [
                x - THUMBNAIL_POINT_RADIUS,
                y - THUMBNAIL_POINT_RADIUS,
                x + THUMBNAIL_POINT_RADIUS,
                y + THUMBNAIL_POINT_RADIUS,
            ],
            fill=color,
            outline="white",
            width=1,
        )

    return pil_image.convert("RGB")


def _render_navigation_controls(images: list[dict], key_suffix: str = "") -> None:
    """Render previous/next navigation buttons."""
    nav_cols = st.columns([1, 3, 1])

    with nav_cols[0]:
        if st.button(
            "Previous",
            key=f"point_prev_{key_suffix}" if key_suffix else None,
            disabled=st.session_state.point_selected_image_index == 0,
        ):
            st.session_state.point_selected_image_index -= 1
            st.rerun()

    with nav_cols[1]:
        st.caption(f"Image {st.session_state.point_selected_image_index + 1} of {len(images)}")

    with nav_cols[2]:
        if st.button(
            "Next",
            key=f"point_next_{key_suffix}" if key_suffix else None,
            disabled=st.session_state.point_selected_image_index >= len(images) - 1,
        ):
            st.session_state.point_selected_image_index += 1
            st.rerun()


def _render_thumbnail_gallery(images: list[dict]) -> None:
    """Render the thumbnail gallery for image selection."""
    st.subheader("Select Image")

    def handle_select(image: dict) -> None:
        idx = next(i for i, img in enumerate(images) if img["id"] == image["id"])
        st.session_state.point_selected_image_index = idx
        st.rerun()

    image_gallery(
        images,
        config=GalleryConfig(
            columns=6,
            max_images=12,
            show_dimensions=False,
            key_prefix="point_annotation_",
            selected_index=st.session_state.point_selected_image_index,
        ),
        on_select=handle_select,
        image_renderer=_create_point_overlay,
    )


def _render_interaction_mode_sidebar() -> str:
    """Render interaction mode controls in sidebar.

    Returns the current interaction mode: 'add' or 'delete'.
    """
    st.subheader("Interaction Mode")

    mode = st.radio(
        "Mode",
        options=["Add Points", "Delete Points"],
        index=0 if st.session_state.point_interaction_mode == "add" else 1,
        key="point_mode_radio",
        label_visibility="collapsed",
    )

    new_mode = "add" if mode == "Add Points" else "delete"
    if new_mode != st.session_state.point_interaction_mode:
        st.session_state.point_interaction_mode = new_mode

    return st.session_state.point_interaction_mode


def _render_point_type_sidebar() -> bool:
    """Render point type controls in sidebar.

    Returns True for positive, False for negative.
    """
    st.subheader("Point Type")

    point_type = st.radio(
        "Type",
        options=["+ Positive (foreground)", "- Negative (background)"],
        index=0 if st.session_state.point_is_positive else 1,
        key="point_type_radio",
        label_visibility="collapsed",
    )

    is_positive = point_type == "+ Positive (foreground)"
    if is_positive != st.session_state.point_is_positive:
        st.session_state.point_is_positive = is_positive

    return st.session_state.point_is_positive


def _render_point_stats(points: list[dict]) -> None:
    """Render point count statistics."""
    st.subheader("Points")

    positive_count = sum(1 for p in points if p["is_positive"])
    negative_count = len(points) - positive_count

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"<div style='display: flex; align-items: center; gap: 6px;'>"
            f"<div style='width: 10px; height: 10px; background: {COLOR_POSITIVE_EXEMPLAR}; "
            f"border-radius: 50%;'></div>"
            f"<span>{positive_count} positive</span></div>",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"<div style='display: flex; align-items: center; gap: 6px;'>"
            f"<div style='width: 10px; height: 10px; background: {COLOR_NEGATIVE_EXEMPLAR}; "
            f"border-radius: 50%;'></div>"
            f"<span>{negative_count} negative</span></div>",
            unsafe_allow_html=True,
        )


def _render_action_buttons(_image_id: str, points: list[dict]) -> None:
    """Render process and clear buttons."""
    st.divider()

    # Process button - will be implemented in Phase 4
    # _image_id will be used when processing is implemented
    if st.button("Process", key="point_process_btn", disabled=len(points) == 0):
        st.info("Processing will be implemented in Phase 4")

    # Clear all points button
    if st.button("Clear All Points", key="point_clear_btn", disabled=len(points) == 0):
        for point in points:
            delete_point_annotation(point["id"])
        st.rerun()


def _render_instructions() -> None:
    """Render the instructions panel."""
    st.markdown("---")
    st.caption("**Instructions:**")
    st.caption("In Add mode: click on the image to place points.")
    st.caption("In Delete mode: click near a point to remove it.")
    st.caption("Green = positive (foreground), Red = negative (background).")


def _render_image_annotator(
    current_image: dict,
    points: list[dict],
    interaction_mode: str,
    is_positive: bool,
) -> None:
    """Render the image with point annotator component."""
    image_id = current_image["id"]
    image_data = fetch_image_data(image_id)

    if not image_data:
        st.error("Failed to load image")
        return

    pil_image = Image.open(BytesIO(image_data))
    st.subheader(current_image["filename"])

    click = point_annotator(
        pil_image,
        points,
        key=f"point_annotator_{image_id}_{interaction_mode}",
    )

    if click:
        if interaction_mode == "add":
            # Create new point annotation
            result = create_point_annotation(image_id, click["x"], click["y"], is_positive)
            if result:
                st.rerun()
            else:
                st.error("Failed to create point annotation")
        else:
            # Delete mode - find and delete nearest point
            nearest = find_point_at_click(click["x"], click["y"], points)
            if nearest:
                if delete_point_annotation(nearest["id"]):
                    st.rerun()
                else:
                    st.error("Failed to delete point annotation")


def _init_session_state() -> None:
    """Initialize session state for point annotation page."""
    if "point_selected_image_index" not in st.session_state:
        st.session_state.point_selected_image_index = 0
    if "point_interaction_mode" not in st.session_state:
        st.session_state.point_interaction_mode = "add"
    if "point_is_positive" not in st.session_state:
        st.session_state.point_is_positive = True


def render() -> None:
    """Render the point annotation page."""
    _init_session_state()

    st.header("Point Annotation")
    st.caption("Click on images to place positive and negative point prompts")

    images = fetch_images()

    if not images:
        st.info("No images uploaded. Please upload images first.")
        return

    # Ensure index is within bounds
    if st.session_state.point_selected_image_index >= len(images):
        st.session_state.point_selected_image_index = 0

    current_image = images[st.session_state.point_selected_image_index]
    points = fetch_point_annotations(current_image["id"])

    # Layout: main content left, sidebar right
    main_col, sidebar_col = st.columns([3, 1])

    # Sidebar controls
    with sidebar_col:
        interaction_mode = _render_interaction_mode_sidebar()

        # Only show point type toggle in add mode
        if interaction_mode == "add":
            st.divider()
            is_positive = _render_point_type_sidebar()
        else:
            is_positive = st.session_state.point_is_positive

        st.divider()
        _render_point_stats(points)
        _render_action_buttons(current_image["id"], points)
        _render_instructions()

    # Main content
    with main_col:
        _render_navigation_controls(images, key_suffix="top")
        _render_image_annotator(current_image, points, interaction_mode, is_positive)
        _render_navigation_controls(images, key_suffix="bottom")
        _render_thumbnail_gallery(images)
