"""Annotation page for drawing bounding boxes on images."""

from io import BytesIO

import streamlit as st
from PIL import Image, ImageDraw

from samui_frontend.api import (
    create_annotation,
    delete_annotation,
    fetch_annotations,
    fetch_image_data,
    fetch_images,
    update_image_text_prompt,
)
from samui_frontend.components.bbox_annotator import bbox_annotator
from samui_frontend.components.image_gallery import GalleryConfig, image_gallery
from samui_frontend.components.mode_toggle import render_mode_toggle
from samui_frontend.components.process_controls import (
    get_images_ready_for_mode,
    render_process_buttons,
    render_processing_status,
)
from samui_frontend.constants import COLOR_NEGATIVE_EXEMPLAR, COLOR_POSITIVE_EXEMPLAR
from samui_frontend.models import PromptType, SegmentationMode
from samui_frontend.utils import get_annotation_color, get_text_prompt_label


def _create_bbox_overlay(image: dict, image_data: bytes) -> Image.Image:
    """Create overlay with bboxes for gallery display."""
    pil_image = Image.open(BytesIO(image_data)).convert("RGBA")

    # Get current mode from session state
    mode = st.session_state.get("segmentation_mode", SegmentationMode.INSIDE_BOX)
    annotations = fetch_annotations(image["id"], mode)

    if not annotations:
        return pil_image.convert("RGB")

    draw = ImageDraw.Draw(pil_image)
    for idx, ann in enumerate(annotations):
        color = get_annotation_color(ann, idx)
        x1 = ann["bbox_x"]
        y1 = ann["bbox_y"]
        x2 = x1 + ann["bbox_width"]
        y2 = y1 + ann["bbox_height"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

    return pil_image.convert("RGB")


def _render_image_annotator(
    current_image: dict,
    annotations: list[dict],
    mode: SegmentationMode,
    exemplar_type: PromptType,
) -> None:
    """Render the image with bbox annotator component."""
    image_id = current_image["id"]
    image_data = fetch_image_data(image_id)

    if not image_data:
        st.error("Failed to load image")
        return

    pil_image = Image.open(BytesIO(image_data))
    st.subheader(current_image["filename"])

    new_bbox = bbox_annotator(pil_image, annotations, key=f"annotator_{image_id}_{mode.value}")

    if new_bbox:
        # Use the passed exemplar_type (SEGMENT for inside_box, selected type for find_all)
        prompt_type = exemplar_type

        if create_annotation(
            image_id,
            new_bbox["x"],
            new_bbox["y"],
            new_bbox["width"],
            new_bbox["height"],
            prompt_type,
        ):
            st.success("Annotation created!")
            st.rerun()
        else:
            st.error("Failed to create annotation")


def _render_navigation_controls(images: list[dict], key_suffix: str = "") -> None:
    """Render previous/next navigation buttons.

    Args:
        images: List of image dicts.
        key_suffix: Suffix for widget keys to allow multiple instances.
    """
    nav_cols = st.columns([1, 3, 1])

    with nav_cols[0]:
        if st.button(
            "Previous",
            key=f"prev_{key_suffix}" if key_suffix else None,
            disabled=st.session_state.selected_image_index == 0,
        ):
            st.session_state.selected_image_index -= 1
            st.rerun()

    with nav_cols[1]:
        st.caption(f"Image {st.session_state.selected_image_index + 1} of {len(images)}")

    with nav_cols[2]:
        if st.button(
            "Next",
            key=f"next_{key_suffix}" if key_suffix else None,
            disabled=st.session_state.selected_image_index >= len(images) - 1,
        ):
            st.session_state.selected_image_index += 1
            st.rerun()


def _render_thumbnail_gallery(images: list[dict]) -> None:
    """Render the thumbnail gallery for image selection."""
    st.subheader("Select Image")

    def handle_select(image: dict) -> None:
        idx = next(i for i, img in enumerate(images) if img["id"] == image["id"])
        st.session_state.selected_image_index = idx
        st.rerun()

    # Show text prompt label in find-all mode
    mode = st.session_state.get("segmentation_mode", SegmentationMode.INSIDE_BOX)
    label_cb = get_text_prompt_label if mode == SegmentationMode.FIND_ALL else None

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
        label_callback=label_cb,
    )


def _render_annotation_list(annotations: list[dict], mode: SegmentationMode) -> None:
    """Render the annotation list sidebar."""
    if mode == SegmentationMode.INSIDE_BOX:
        header = f"Boxes ({len(annotations)})"
        empty_msg = "No boxes yet. Draw a bounding box on the image."
    else:
        header = f"Exemplars ({len(annotations)})"
        empty_msg = "No exemplars yet. Draw positive (left-click) or negative (right-click) exemplars."

    st.subheader(header)

    if not annotations:
        st.info(empty_msg)
        return

    for idx, annotation in enumerate(annotations):
        color = get_annotation_color(annotation, idx)
        prompt_type = annotation.get("prompt_type", PromptType.SEGMENT.value)

        # Build label based on prompt type
        if prompt_type == PromptType.POSITIVE_EXEMPLAR.value:
            label = f"+ Exemplar {idx + 1}"
            badge_color = COLOR_POSITIVE_EXEMPLAR
        elif prompt_type == PromptType.NEGATIVE_EXEMPLAR.value:
            label = f"- Exemplar {idx + 1}"
            badge_color = COLOR_NEGATIVE_EXEMPLAR
        else:
            label = f"Box {idx + 1}"
            badge_color = color

        col1, col2 = st.columns([4, 1])

        with col1:
            st.markdown(
                f"<div style='display: flex; align-items: center;'>"
                f"<div style='width: 12px; height: 12px; background: {badge_color}; "
                f"border-radius: 3px; margin-right: 8px;'></div>"
                f"<span>{label}</span></div>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"x: {annotation['bbox_x']}, y: {annotation['bbox_y']}, "
                f"w: {annotation['bbox_width']}, h: {annotation['bbox_height']}"
            )

        with col2:
            if st.button("X", key=f"del_{annotation['id']}"):
                if delete_annotation(annotation["id"]):
                    st.rerun()
                else:
                    st.error("Failed to delete")

        st.divider()


def _render_instructions(mode: SegmentationMode) -> None:
    """Render the instructions panel."""
    st.markdown("---")
    st.caption("**Instructions:**")
    if mode == SegmentationMode.INSIDE_BOX:
        st.caption("Click and drag on the image to draw bounding boxes.")
        st.caption("Use Previous/Next buttons to navigate between images.")
    else:
        st.caption("Enter a text prompt describing what to find.")
        st.caption("Select exemplar type (+ or -) then draw boxes.")
        st.caption("Positive (+): find similar objects.")
        st.caption("Negative (-): exclude similar objects.")


def _render_find_all_controls(current_image: dict) -> PromptType:
    """Render text prompt input and exemplar type toggle for find-all mode.

    Returns the selected exemplar type for creating new annotations.
    """
    # Display current text prompt with delete button
    current_prompt = current_image.get("text_prompt") or ""
    if current_prompt:
        prompt_col, delete_col = st.columns([4, 1])
        with prompt_col:
            st.info(f"**Text prompt:** {current_prompt}")
        with delete_col:
            if st.button("X", key=f"clear_prompt_{current_image['id']}", help="Clear text prompt"):
                if update_image_text_prompt(current_image["id"], None):
                    st.rerun()
                else:
                    st.error("Failed to clear")

    col1, col2 = st.columns([2, 1])

    with col1:
        current_prompt = current_image.get("text_prompt", "") or ""
        text_key = f"text_prompt_{current_image['id']}"

        new_prompt = st.text_input(
            "Text Prompt",
            value=current_prompt,
            placeholder="e.g., 'red apples' or 'person wearing blue'",
            key=text_key,
            help="Describe what you want to find in the image",
        )

        # Save if changed
        if new_prompt != current_prompt:
            if update_image_text_prompt(current_image["id"], new_prompt):
                st.rerun()
            else:
                st.error("Failed to save text prompt")

    with col2:
        exemplar_options = ["+ Positive", "- Negative"]
        current_idx = 0 if st.session_state.exemplar_type == PromptType.POSITIVE_EXEMPLAR else 1

        selected = st.radio(
            "Exemplar Type",
            options=exemplar_options,
            index=current_idx,
            horizontal=True,
            key="exemplar_type_radio",
            help="Positive: find similar. Negative: exclude similar.",
        )

        # Map selection back to enum
        new_exemplar_type = PromptType.POSITIVE_EXEMPLAR if selected == "+ Positive" else PromptType.NEGATIVE_EXEMPLAR

        # Update session state if changed
        if new_exemplar_type != st.session_state.exemplar_type:
            st.session_state.exemplar_type = new_exemplar_type

    return st.session_state.exemplar_type


def render() -> None:
    """Render the annotation page."""
    st.header("Annotate Images")
    st.caption("Draw bounding boxes to define segmentation regions")

    images = fetch_images()

    # Mode toggle and process controls in same row
    mode_col, process_col, status_col = st.columns([1, 1, 1])

    with mode_col:
        current_mode = render_mode_toggle(key="mode_radio")

    with process_col:
        ready_images = get_images_ready_for_mode(images, current_mode) if images else []
        render_process_buttons(ready_images, current_mode)

    with status_col:
        render_processing_status()

    st.divider()

    if not images:
        st.info("No images uploaded. Please upload images first.")
        return

    if st.session_state.selected_image_index >= len(images):
        st.session_state.selected_image_index = 0

    current_image = images[st.session_state.selected_image_index]
    annotations = fetch_annotations(current_image["id"], current_mode)

    # Show find-all controls (text prompt + exemplar type toggle)
    exemplar_type = PromptType.SEGMENT  # Default for inside_box mode
    if current_mode == SegmentationMode.FIND_ALL:
        exemplar_type = _render_find_all_controls(current_image)

    main_col, sidebar_col = st.columns([3, 1])

    with main_col:
        _render_navigation_controls(images, key_suffix="top")
        _render_image_annotator(current_image, annotations, current_mode, exemplar_type)
        _render_navigation_controls(images, key_suffix="bottom")
        _render_thumbnail_gallery(images)

    with sidebar_col:
        _render_annotation_list(annotations, current_mode)
        _render_instructions(current_mode)
