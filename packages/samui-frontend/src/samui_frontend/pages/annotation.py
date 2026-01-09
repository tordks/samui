"""Annotation page for drawing bounding boxes on images."""

from io import BytesIO

import httpx
import streamlit as st
from PIL import Image, ImageDraw

from samui_frontend.components.bbox_annotator import bbox_annotator, get_bbox_color
from samui_frontend.components.image_gallery import GalleryConfig, image_gallery
from samui_frontend.config import API_URL

# Color palette for bounding boxes (matches bbox_annotator)
BBOX_COLORS = [
    "#ff4b4b",  # red
    "#4bff4b",  # green
    "#4b4bff",  # blue
    "#ffff4b",  # yellow
    "#ff4bff",  # magenta
    "#4bffff",  # cyan
]


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


def _create_bbox_overlay(image: dict, image_data: bytes) -> Image.Image:
    """Create overlay with bboxes for gallery display."""
    pil_image = Image.open(BytesIO(image_data)).convert("RGBA")
    annotations = _fetch_annotations(image["id"])

    if not annotations:
        return pil_image.convert("RGB")

    draw = ImageDraw.Draw(pil_image)
    for idx, ann in enumerate(annotations):
        color = BBOX_COLORS[idx % len(BBOX_COLORS)]
        x1 = ann["bbox_x"]
        y1 = ann["bbox_y"]
        x2 = x1 + ann["bbox_width"]
        y2 = y1 + ann["bbox_height"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

    return pil_image.convert("RGB")


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

    def handle_select(image: dict) -> None:
        idx = next(i for i, img in enumerate(images) if img["id"] == image["id"])
        st.session_state.selected_image_index = idx
        st.rerun()

    image_gallery(
        images,
        config=GalleryConfig(
            columns=6,
            max_images=12,
            show_dimensions=False,
            key_prefix="annotation_",
            selected_index=st.session_state.selected_image_index,
        ),
        on_select=handle_select,
        image_renderer=_create_bbox_overlay,
    )


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
