"""Processing page for running SAM3 inference and viewing results."""

import io
import time

import httpx
import streamlit as st
from PIL import Image, ImageDraw

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


def _fetch_mask_data(image_id: str) -> bytes | None:
    """Fetch mask data from the API."""
    try:
        response = httpx.get(f"{API_URL}/process/mask/{image_id}", timeout=10.0)
        if response.status_code == 200:
            return response.content
    except httpx.HTTPError:
        pass
    return None


def _fetch_annotations(image_id: str) -> list[dict]:
    """Fetch annotations for an image."""
    try:
        response = httpx.get(f"{API_URL}/annotations/{image_id}", timeout=10.0)
        if response.status_code == 200:
            return response.json().get("annotations", [])
    except httpx.HTTPError:
        pass
    return []


def _create_overlay_image(
    image_data: bytes,
    mask_data: bytes | None,
    annotations: list[dict],
) -> Image.Image:
    """Create an image with bbox and mask overlays.

    Args:
        image_data: Original image bytes.
        mask_data: Mask image bytes (grayscale PNG) or None.
        annotations: List of annotation dicts with bbox coordinates.

    Returns:
        PIL Image with overlays applied.
    """
    # Load the original image
    image = Image.open(io.BytesIO(image_data)).convert("RGBA")
    draw = ImageDraw.Draw(image)

    # Draw bounding boxes
    for idx, ann in enumerate(annotations):
        color = BBOX_COLORS[idx % len(BBOX_COLORS)]
        x1 = ann["bbox_x"]
        y1 = ann["bbox_y"]
        x2 = x1 + ann["bbox_width"]
        y2 = y1 + ann["bbox_height"]

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label
        label = f"Box {idx + 1}"
        label_bbox = draw.textbbox((x1, y1 - 20), label)
        draw.rectangle(label_bbox, fill=color)
        draw.text((x1, y1 - 20), label, fill="white")

    # Overlay mask if available
    if mask_data:
        mask = Image.open(io.BytesIO(mask_data)).convert("L")

        # Resize mask to match image if needed
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.NEAREST)

        # Where mask is white (255), overlay semi-transparent green
        mask_rgba = Image.new("RGBA", image.size, (0, 255, 0, 100))
        image = Image.composite(mask_rgba, image, mask)

    return image.convert("RGB")


def _create_gallery_overlay(image: dict, image_data: bytes) -> Image.Image:
    """Create overlay image for gallery display.

    Shows bbox+segmentation for processed images, just bbox for annotated,
    and raw image for pending.
    """
    status = image.get("processing_status")
    annotations = _fetch_annotations(image["id"])
    mask_data = _fetch_mask_data(image["id"]) if status == "processed" else None
    return _create_overlay_image(image_data, mask_data, annotations)


def _start_processing(image_ids: list[str]) -> dict | None:
    """Start processing for given image IDs."""
    try:
        response = httpx.post(
            f"{API_URL}/process",
            json={"image_ids": image_ids},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        st.error(f"Failed to start processing: {e}")
        return None


def _get_processing_status() -> dict | None:
    """Get current processing status."""
    try:
        response = httpx.get(f"{API_URL}/process/status", timeout=10.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


def _download_coco_json(image_id: str) -> dict | None:
    """Download COCO JSON for an image."""
    try:
        response = httpx.get(f"{API_URL}/process/export/{image_id}", timeout=30.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


def _download_all_coco_json(processed_images: list[dict]) -> dict | None:
    """Download combined COCO JSON for all processed images.

    Fetches individual COCO data and combines into single file with:
    - Combined images list
    - Combined annotations list (with updated IDs to avoid conflicts)
    - Single categories list
    """
    if not processed_images:
        return None

    combined = {
        "images": [],
        "annotations": [],
        "categories": [],
    }
    categories_seen = set()
    annotation_id_offset = 0

    for img in processed_images:
        coco_data = _download_coco_json(img["id"])
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


def _render_process_controls(
    annotated_images: list[dict],
    processed_images: list[dict],
) -> None:
    """Render the process button, download button, and progress indicator."""
    import json

    st.subheader("Processing Controls")

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        process_disabled = len(annotated_images) == 0
        if st.button(
            "Process All Annotated",
            disabled=process_disabled,
            type="primary",
        ):
            image_ids = [img["id"] for img in annotated_images]
            result = _start_processing(image_ids)
            if result:
                st.session_state.processing_batch_id = result.get("batch_id")
                st.success(f"Processing started for {result.get('total_images')} images")
                st.rerun()

    with col2:
        if processed_images:
            coco_data = _download_all_coco_json(processed_images)
            if coco_data:
                st.download_button(
                    "Download All COCO",
                    data=json.dumps(coco_data, indent=2),
                    file_name="coco_annotations.json",
                    mime="application/json",
                )
        else:
            st.button("Download All COCO", disabled=True)

    with col3:
        status = _get_processing_status()
        if status and status.get("is_running"):
            processed = status.get("processed_count", 0)
            total = status.get("total_count", 0)
            current_filename = status.get("current_image_filename", "")

            progress = processed / total if total > 0 else 0
            st.progress(progress, f"Processing: {processed} of {total} ({current_filename})")

            # Auto-refresh while processing
            time.sleep(1)
            st.rerun()
        elif status and status.get("error"):
            st.error(f"Processing failed: {status.get('error')}")
        elif status and status.get("batch_id") and status.get("processed_count", 0) > 0:
            st.success(f"Processing complete! {status.get('processed_count')} images processed.")
        elif len(annotated_images) == 0:
            st.info("No annotated images to process. Add annotations first.")
        else:
            st.info(f"{len(annotated_images)} images ready for processing.")


def _render_processed_gallery(images: list[dict]) -> None:
    """Render the gallery of all images with overlays."""
    st.subheader(f"Images ({len(images)})")

    if not images:
        return

    def handle_select(image: dict) -> None:
        idx = next(i for i, img in enumerate(images) if img["id"] == image["id"])
        st.session_state.selected_processed_index = idx
        st.rerun()

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
    )


def render() -> None:
    """Render the processing page."""
    st.header("Process Images")
    st.caption("Run SAM3 inference on annotated images")

    # Fetch all images
    images = _fetch_images()

    # Filter by status
    annotated_images = [
        img for img in images if img.get("processing_status") in ["annotated", "processing"]
    ]
    processed_images = [
        img for img in images if img.get("processing_status") == "processed"
    ]

    # Render processing controls
    _render_process_controls(annotated_images, processed_images)

    st.divider()

    # Render image gallery
    _render_processed_gallery(images)
