"""Processing page for running SAM3 inference and viewing results."""

import io
import json
import time

import streamlit as st
from PIL import Image, ImageDraw

from samui_frontend.api import (
    download_coco_json,
    fetch_annotations,
    fetch_images,
    fetch_mask_data,
    get_processing_status,
    start_processing,
)
from samui_frontend.components.image_gallery import GalleryConfig, image_gallery
from samui_frontend.constants import (
    BBOX_COLORS,
    COLOR_NEGATIVE_EXEMPLAR,
    COLOR_POSITIVE_EXEMPLAR,
)
from samui_frontend.models import PromptType, SegmentationMode


def _get_annotation_color(annotation: dict, index: int) -> str:
    """Get color for annotation based on prompt_type."""
    prompt_type = annotation.get("prompt_type", PromptType.SEGMENT.value)
    if prompt_type == PromptType.POSITIVE_EXEMPLAR.value:
        return COLOR_POSITIVE_EXEMPLAR
    elif prompt_type == PromptType.NEGATIVE_EXEMPLAR.value:
        return COLOR_NEGATIVE_EXEMPLAR
    return BBOX_COLORS[index % len(BBOX_COLORS)]


def _get_annotation_label(annotation: dict, index: int, mode: SegmentationMode) -> str:
    """Get label for annotation based on prompt_type and mode."""
    prompt_type = annotation.get("prompt_type", PromptType.SEGMENT.value)
    if mode == SegmentationMode.FIND_ALL:
        if prompt_type == PromptType.POSITIVE_EXEMPLAR.value:
            return f"+Ex {index + 1}"
        elif prompt_type == PromptType.NEGATIVE_EXEMPLAR.value:
            return f"-Ex {index + 1}"
    return f"Box {index + 1}"


def _get_text_prompt_label(image: dict) -> str | None:
    """Return text prompt label for gallery display."""
    text_prompt = image.get("text_prompt")
    if text_prompt:
        return f"text prompt: {text_prompt}"
    return None


def _create_overlay_image(
    image_data: bytes,
    mask_data: bytes | None,
    annotations: list[dict],
    mode: SegmentationMode = SegmentationMode.INSIDE_BOX,
) -> Image.Image:
    """Create an image with bbox and mask overlays.

    Args:
        image_data: Original image bytes.
        mask_data: Mask image bytes (grayscale PNG) or None.
        annotations: List of annotation dicts with bbox coordinates.
        mode: Segmentation mode for label formatting.

    Returns:
        PIL Image with overlays applied.
    """
    # Load the original image
    image = Image.open(io.BytesIO(image_data)).convert("RGBA")

    # Overlay mask first (so bboxes appear on top)
    if mask_data:
        mask = Image.open(io.BytesIO(mask_data)).convert("L")

        # Resize mask to match image if needed
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.NEAREST)

        # Where mask is white (255), overlay semi-transparent green
        mask_rgba = Image.new("RGBA", image.size, (0, 255, 0, 100))
        image = Image.composite(mask_rgba, image, mask)

    # Draw bounding boxes on top
    draw = ImageDraw.Draw(image)
    for idx, ann in enumerate(annotations):
        color = _get_annotation_color(ann, idx)
        x1 = ann["bbox_x"]
        y1 = ann["bbox_y"]
        x2 = x1 + ann["bbox_width"]
        y2 = y1 + ann["bbox_height"]

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label
        label = _get_annotation_label(ann, idx, mode)
        label_bbox = draw.textbbox((x1, y1 - 20), label)
        draw.rectangle(label_bbox, fill=color)
        draw.text((x1, y1 - 20), label, fill="white")

    return image.convert("RGB")


def _create_gallery_overlay(image: dict, image_data: bytes) -> Image.Image:
    """Create overlay image for gallery display.

    Shows bbox+segmentation for processed images, just bbox for annotated,
    and raw image for pending. Uses current segmentation mode from session state.
    """
    mode = st.session_state.get("segmentation_mode", SegmentationMode.INSIDE_BOX)
    annotations = fetch_annotations(image["id"], mode, for_display=True)

    # Check if image has been processed for this mode
    # For now, we only show mask if we have annotations for the mode
    # The mask endpoint handles mode-specific results
    mask_data = None
    if annotations:
        mask_data = fetch_mask_data(image["id"], mode)

    return _create_overlay_image(image_data, mask_data, annotations, mode)


def _download_all_coco_json(processed_images: list[dict], mode: SegmentationMode | None = None) -> dict | None:
    """Download combined COCO JSON for all processed images.

    Fetches individual COCO data and combines into single file with:
    - Combined images list
    - Combined annotations list (with updated IDs to avoid conflicts)
    - Single categories list
    """
    if not processed_images:
        return None

    combined: dict = {
        "images": [],
        "annotations": [],
        "categories": [],
    }
    categories_seen: set[int] = set()
    annotation_id_offset = 0

    for img in processed_images:
        coco_data = download_coco_json(img["id"], mode)
        if not coco_data:
            continue

        # Add images
        combined["images"].extend(coco_data.get("images", []))

        # Add annotations with offset IDs
        for ann in coco_data.get("annotations", []):
            ann_copy = ann.copy()
            ann_copy["id"] = ann_copy["id"] + annotation_id_offset
            combined["annotations"].append(ann_copy)

        # Track max annotation ID for offset
        if coco_data.get("annotations"):
            max_id = max(a["id"] for a in coco_data["annotations"])
            annotation_id_offset += max_id + 1

        # Add unique categories
        for cat in coco_data.get("categories", []):
            if cat["id"] not in categories_seen:
                categories_seen.add(cat["id"])
                combined["categories"].append(cat)

    return combined if combined["images"] else None


def _get_images_ready_for_mode(images: list[dict], mode: SegmentationMode) -> list[dict]:
    """Filter images that are ready for processing in the given mode.

    For inside_box: images with SEGMENT annotations
    For find_all: images with text_prompt OR exemplar annotations
    """
    ready = []
    for img in images:
        if mode == SegmentationMode.INSIDE_BOX:
            # Check for segment annotations
            annotations = fetch_annotations(img["id"], mode)
            if annotations:
                ready.append(img)
        else:
            # Check for text_prompt or exemplar annotations
            has_text = bool(img.get("text_prompt"))
            annotations = fetch_annotations(img["id"], mode)
            if has_text or annotations:
                ready.append(img)
    return ready


def _get_annotation_badge(image: dict, mode: SegmentationMode) -> str:
    """Get a badge string describing annotations for an image in the given mode."""
    annotations = fetch_annotations(image["id"], mode)

    if mode == SegmentationMode.INSIDE_BOX:
        count = len(annotations)
        return f"{count} box{'es' if count != 1 else ''}" if count > 0 else "No boxes"
    else:
        # Find all mode
        has_text = bool(image.get("text_prompt"))
        pos_count = sum(1 for a in annotations if a.get("prompt_type") == PromptType.POSITIVE_EXEMPLAR.value)
        neg_count = sum(1 for a in annotations if a.get("prompt_type") == PromptType.NEGATIVE_EXEMPLAR.value)

        parts = []
        if has_text:
            parts.append("text")
        if pos_count:
            parts.append(f"+{pos_count}")
        if neg_count:
            parts.append(f"-{neg_count}")

        return ", ".join(parts) if parts else "No prompts"


def _render_mode_toggle() -> SegmentationMode:
    """Render the segmentation mode toggle and return the selected mode."""
    mode_options = {
        SegmentationMode.INSIDE_BOX: "Inside Box",
        SegmentationMode.FIND_ALL: "Find All",
    }
    mode_descriptions = {
        SegmentationMode.INSIDE_BOX: "Segment objects inside each bounding box",
        SegmentationMode.FIND_ALL: "Find all instances matching text prompt or exemplars",
    }

    col1, col2 = st.columns([1, 2])
    with col1:
        selected_label = st.radio(
            "Segmentation Mode",
            options=list(mode_options.values()),
            index=0 if st.session_state.segmentation_mode == SegmentationMode.INSIDE_BOX else 1,
            horizontal=True,
            key="processing_mode_radio",
        )

    # Map label back to enum
    mode = SegmentationMode.INSIDE_BOX if selected_label == "Inside Box" else SegmentationMode.FIND_ALL

    # Update session state if mode changed
    if mode != st.session_state.segmentation_mode:
        st.session_state.segmentation_mode = mode
        st.rerun()

    with col2:
        st.caption(mode_descriptions[mode])

    return mode


def _render_process_controls(
    ready_images: list[dict],
    processed_images: list[dict],
    mode: SegmentationMode,
) -> None:
    """Render the process button, download button, and progress indicator."""

    mode_label = "Inside Box" if mode == SegmentationMode.INSIDE_BOX else "Find All"
    st.subheader("Processing Controls")

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        process_disabled = len(ready_images) == 0
        button_label = f"Process ({mode_label})"
        if st.button(
            button_label,
            disabled=process_disabled,
            type="primary",
        ):
            image_ids = [img["id"] for img in ready_images]
            result = start_processing(image_ids, mode)
            if result:
                st.session_state.processing_batch_id = result.get("batch_id")
                st.success(f"Processing started for {result.get('total_images')} images")
                st.rerun()
            else:
                st.error("Failed to start processing")

    with col2:
        if processed_images:
            coco_data = _download_all_coco_json(processed_images, mode)
            if coco_data:
                st.download_button(
                    "Download All COCO",
                    data=json.dumps(coco_data, indent=2),
                    file_name=f"coco_annotations_{mode.value}.json",
                    mime="application/json",
                )
        else:
            st.button("Download All COCO", disabled=True)

    with col3:
        status = get_processing_status()
        if status and status.get("is_running"):
            processed = status.get("processed_count", 0)
            total = status.get("total_count", 0)
            current_filename = status.get("current_image_filename", "")

            progress = processed / total if total > 0 else 0
            st.progress(progress, f"Processing ({mode_label}): {processed} of {total} ({current_filename})")

            # Auto-refresh while processing
            time.sleep(1)
            st.rerun()
        elif status and status.get("error"):
            st.error(f"Processing failed: {status.get('error')}")
        elif status and status.get("batch_id") and status.get("processed_count", 0) > 0:
            st.success(f"Processing complete! {status.get('processed_count')} images processed.")
        elif len(ready_images) == 0:
            if mode == SegmentationMode.INSIDE_BOX:
                st.info("No images with segment boxes. Add annotations first.")
            else:
                st.info("No images with text prompts or exemplars. Add prompts first.")
        else:
            st.info(f"{len(ready_images)} images ready for {mode_label} processing.")


def _render_processed_gallery(images: list[dict], mode: SegmentationMode) -> None:
    """Render the gallery of all images with overlays."""
    # Count images with annotations for this mode
    ready_count = len(_get_images_ready_for_mode(images, mode))
    mode_label = "Inside Box" if mode == SegmentationMode.INSIDE_BOX else "Find All"

    st.subheader(f"Images ({len(images)}) - {ready_count} ready for {mode_label}")

    if not images:
        return

    def handle_select(image: dict) -> None:
        idx = next(i for i, img in enumerate(images) if img["id"] == image["id"])
        st.session_state.selected_processed_index = idx
        st.rerun()

    # Show text prompt label above thumbnails in find-all mode
    label_cb = _get_text_prompt_label if mode == SegmentationMode.FIND_ALL else None

    image_gallery(
        images,
        config=GalleryConfig(
            columns=4,
            max_images=12,
            show_dimensions=False,
            key_prefix="processed_",
            selected_index=st.session_state.get("selected_processed_index", 0),
        ),
        on_select=handle_select,
        image_renderer=_create_gallery_overlay,
        label_callback=label_cb,
    )

    # Show annotation summary for selected image
    if images:
        selected_idx = st.session_state.get("selected_processed_index", 0)
        if selected_idx < len(images):
            selected_image = images[selected_idx]
            badge = _get_annotation_badge(selected_image, mode)
            st.caption(f"Selected: {selected_image['filename']} - {badge}")


def render() -> None:
    """Render the processing page."""
    st.header("Process Images")
    st.caption("Run SAM3 inference on annotated images")

    # Initialize session state
    if "segmentation_mode" not in st.session_state:
        st.session_state.segmentation_mode = SegmentationMode.INSIDE_BOX

    # Mode toggle at the top
    current_mode = _render_mode_toggle()
    st.divider()

    # Fetch all images
    images = fetch_images()

    # Get images ready for processing in current mode
    ready_images = _get_images_ready_for_mode(images, current_mode)

    # For processed images, we still show all - the gallery overlay handles mode-specific display
    processed_images = [img for img in images if img.get("processing_status") == "processed"]

    # Render processing controls with mode
    _render_process_controls(ready_images, processed_images, current_mode)

    st.divider()

    # Render image gallery
    _render_processed_gallery(images, current_mode)
