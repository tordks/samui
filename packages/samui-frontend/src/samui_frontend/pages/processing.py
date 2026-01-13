"""Processing page for running SAM3 inference and viewing results."""

import io
import json

import streamlit as st
from PIL import Image, ImageDraw

from samui_frontend.api import (
    create_job,
    download_coco_json,
    fetch_annotations,
    fetch_images,
    fetch_job,
    fetch_mask_data,
)
from samui_frontend.components.image_gallery import GalleryConfig, image_gallery
from samui_frontend.components.mode_toggle import render_mode_toggle
from samui_frontend.models import PromptType, SegmentationMode
from samui_frontend.utils import (
    get_annotation_color,
    get_annotation_label,
    get_text_prompt_label,
)


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
        color = get_annotation_color(ann, idx)
        x1 = ann["bbox_x"]
        y1 = ann["bbox_y"]
        x2 = x1 + ann["bbox_width"]
        y2 = y1 + ann["bbox_height"]

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label
        label = get_annotation_label(ann, idx, mode)
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
    annotations = fetch_annotations(image["id"], mode)

    # Try to get mask for this mode - will be None if not processed
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


@st.fragment(run_every=1)
def _render_processing_status(mode: SegmentationMode, ready_count: int) -> None:
    """Fragment that polls processing status with auto-refresh.

    Using @st.fragment(run_every=1) allows this component to refresh
    every second without blocking the main UI or causing a full page rerun.

    When processed_count changes, triggers a full rerun to update the gallery.
    """
    mode_label = "Inside Box" if mode == SegmentationMode.INSIDE_BOX else "Find All"
    job_id = st.session_state.get("current_job_id")

    if job_id:
        job = fetch_job(job_id)
        if job and job.get("is_running"):
            processed = job.get("processed_count", 0)
            total = job.get("image_count", 0)
            current_filename = job.get("current_image_filename", "")

            progress = processed / total if total > 0 else 0
            st.progress(progress, f"Processing ({mode_label}): {processed} of {total} ({current_filename})")

            # Trigger full page rerun when an image finishes processing
            # This updates the gallery to show new results
            last_count = st.session_state.get("last_processed_count", 0)
            if processed > last_count:
                st.session_state.last_processed_count = processed
                st.rerun()
        elif job and job.get("error"):
            st.error(f"Processing failed: {job.get('error')}")
            # Reset state on error
            st.session_state.last_processed_count = 0
            st.session_state.current_job_id = None
        elif job and job.get("status") == "completed":
            st.success(f"Processing complete! {job.get('processed_count', 0)} images processed.")
            # Reset state when complete
            st.session_state.last_processed_count = 0
            st.session_state.current_job_id = None
        else:
            # Job not found or unexpected state - clear it
            st.session_state.current_job_id = None
    elif ready_count == 0:
        if mode == SegmentationMode.INSIDE_BOX:
            st.info("No images with segment boxes. Add annotations first.")
        else:
            st.info("No images with text prompts or exemplars. Add prompts first.")
    else:
        st.info(f"{ready_count} images ready for {mode_label} processing.")


def _render_process_controls(
    ready_images: list[dict],
    processed_images: list[dict],
    mode: SegmentationMode,
) -> None:
    """Render the process buttons, download button, and progress indicator."""
    st.subheader("Processing Controls")

    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

    with col1:
        process_disabled = len(ready_images) == 0
        if st.button(
            "Process",
            disabled=process_disabled,
            type="primary",
            help="Process only images with changed annotations",
        ):
            image_ids = [img["id"] for img in ready_images]
            result = create_job(image_ids, mode, force_all=False)
            if result:
                st.session_state.current_job_id = result.get("id")
                st.session_state.last_processed_count = 0
                st.success(f"Job created for {result.get('image_count', 0)} images")
                st.rerun()
            else:
                st.error("Failed to create processing job")

    with col2:
        process_all_disabled = len(ready_images) == 0
        if st.button(
            "Process All",
            disabled=process_all_disabled,
            help="Process all images regardless of changes",
        ):
            image_ids = [img["id"] for img in ready_images]
            result = create_job(image_ids, mode, force_all=True)
            if result:
                st.session_state.current_job_id = result.get("id")
                st.session_state.last_processed_count = 0
                st.success(f"Job created for {result.get('image_count', 0)} images")
                st.rerun()
            else:
                st.error("Failed to create processing job")

    with col3:
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
                st.button("Download All COCO", disabled=True, help="No processed results available")
        else:
            st.button("Download All COCO", disabled=True)

    with col4:
        _render_processing_status(mode, len(ready_images))


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
    label_cb = get_text_prompt_label if mode == SegmentationMode.FIND_ALL else None

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

    # Mode toggle at the top
    current_mode = render_mode_toggle(key="processing_mode_radio")
    st.divider()

    # Fetch all images
    images = fetch_images()

    # Get images ready for processing in current mode
    ready_images = _get_images_ready_for_mode(images, current_mode)

    # Render processing controls with mode
    # For download, use ready_images - the download function handles missing results gracefully
    _render_process_controls(ready_images, ready_images, current_mode)

    st.divider()

    # Render image gallery
    _render_processed_gallery(images, current_mode)
