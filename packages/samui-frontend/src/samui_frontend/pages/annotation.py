"""Annotation page for drawing bounding boxes on images."""

from io import BytesIO

import httpx
from PIL import Image
import streamlit as st

from samui_frontend.components.bbox_annotator import bbox_annotator, get_bbox_color
from samui_frontend.config import API_URL


def _fetch_images() -> list[dict]:
    """Fetch all images from the API."""
    try:
        response = httpx.get(f"{API_URL}/images", timeout=10.0)
        response.raise_for_status()
        return response.json().get("images", [])
    except httpx.HTTPError:
        return []


def _fetch_image_data(image_id: str) -> bytes | None:
    """Fetch image data from the API."""
    try:
        response = httpx.get(f"{API_URL}/images/{image_id}/data", timeout=10.0)
        if response.status_code == 200:
            return response.content
    except httpx.HTTPError:
        pass
    return None


def _fetch_annotations(image_id: str) -> list[dict]:
    """Fetch annotations for an image."""
    try:
        response = httpx.get(f"{API_URL}/annotations/{image_id}", timeout=10.0)
        response.raise_for_status()
        return response.json().get("annotations", [])
    except httpx.HTTPError:
        return []


def _create_annotation(image_id: str, x: int, y: int, width: int, height: int) -> bool:
    """Create a new annotation."""
    try:
        response = httpx.post(
            f"{API_URL}/annotations",
            json={
                "image_id": image_id,
                "bbox_x": x,
                "bbox_y": y,
                "bbox_width": width,
                "bbox_height": height,
            },
            timeout=10.0,
        )
        response.raise_for_status()
        return True
    except httpx.HTTPError:
        return False


def _delete_annotation(annotation_id: str) -> bool:
    """Delete an annotation."""
    try:
        response = httpx.delete(f"{API_URL}/annotations/{annotation_id}", timeout=10.0)
        response.raise_for_status()
        return True
    except httpx.HTTPError:
        return False


def _render_image_annotator(current_image: dict, annotations: list[dict]) -> None:
    """Render the image with bbox annotator component."""
    image_id = current_image["id"]
    image_data = _fetch_image_data(image_id)

    if not image_data:
        st.error("Failed to load image")
        return

    pil_image = Image.open(BytesIO(image_data))
    st.subheader(current_image["filename"])

    new_bbox = bbox_annotator(pil_image, annotations, key=f"annotator_{image_id}")

    if new_bbox:
        if _create_annotation(
            image_id, new_bbox["x"], new_bbox["y"], new_bbox["width"], new_bbox["height"]
        ):
            st.success("Annotation created!")
            st.rerun()
        else:
            st.error("Failed to create annotation")


def _render_navigation_controls(images: list[dict]) -> None:
    """Render previous/next navigation buttons."""
    st.divider()
    nav_cols = st.columns([1, 3, 1])

    with nav_cols[0]:
        if st.button("Previous", disabled=st.session_state.selected_image_index == 0):
            st.session_state.selected_image_index -= 1
            st.rerun()

    with nav_cols[1]:
        st.caption(f"Image {st.session_state.selected_image_index + 1} of {len(images)}")

    with nav_cols[2]:
        if st.button("Next", disabled=st.session_state.selected_image_index >= len(images) - 1):
            st.session_state.selected_image_index += 1
            st.rerun()


def _render_thumbnail_gallery(images: list[dict]) -> None:
    """Render the thumbnail gallery for image selection."""
    st.subheader("Select Image")
    num_cols = min(len(images), 6)
    thumb_cols = st.columns(num_cols)

    for idx, img in enumerate(images[:12]):
        with thumb_cols[idx % num_cols]:
            thumb_data = _fetch_image_data(img["id"])
            if thumb_data:
                is_selected = idx == st.session_state.selected_image_index
                if st.button(
                    "Selected" if is_selected else "Select",
                    key=f"thumb_{img['id']}",
                    disabled=is_selected,
                ):
                    st.session_state.selected_image_index = idx
                    st.rerun()
                st.image(thumb_data, use_container_width=True)


def _render_annotation_list(annotations: list[dict]) -> None:
    """Render the annotation list sidebar."""
    st.subheader(f"Annotations ({len(annotations)})")

    if not annotations:
        st.info("No annotations yet. Draw a bounding box on the image.")
        return

    for idx, annotation in enumerate(annotations):
        color = get_bbox_color(idx)
        col1, col2 = st.columns([4, 1])

        with col1:
            st.markdown(
                f"<div style='display: flex; align-items: center;'>"
                f"<div style='width: 12px; height: 12px; background: {color}; "
                f"border-radius: 3px; margin-right: 8px;'></div>"
                f"<span>Box {idx + 1}</span></div>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"x: {annotation['bbox_x']}, y: {annotation['bbox_y']}, "
                f"w: {annotation['bbox_width']}, h: {annotation['bbox_height']}"
            )

        with col2:
            if st.button("X", key=f"del_{annotation['id']}"):
                if _delete_annotation(annotation["id"]):
                    st.rerun()
                else:
                    st.error("Failed to delete")

        st.divider()


def _render_instructions() -> None:
    """Render the instructions panel."""
    st.markdown("---")
    st.caption("**Instructions:**")
    st.caption("Click and drag on the image to draw bounding boxes.")
    st.caption("Use Previous/Next buttons to navigate between images.")


def render() -> None:
    """Render the annotation page."""
    st.header("Annotate Images")
    st.caption("Draw bounding boxes to define segmentation regions")

    if "selected_image_index" not in st.session_state:
        st.session_state.selected_image_index = 0

    images = _fetch_images()

    if not images:
        st.info("No images uploaded. Please upload images first.")
        return

    if st.session_state.selected_image_index >= len(images):
        st.session_state.selected_image_index = 0

    current_image = images[st.session_state.selected_image_index]
    annotations = _fetch_annotations(current_image["id"])

    main_col, sidebar_col = st.columns([3, 1])

    with main_col:
        _render_image_annotator(current_image, annotations)
        _render_navigation_controls(images)
        _render_thumbnail_gallery(images)

    with sidebar_col:
        _render_annotation_list(annotations)
        _render_instructions()
