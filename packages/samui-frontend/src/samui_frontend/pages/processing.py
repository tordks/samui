"""Processing page for running SAM3 inference and viewing results."""

import io
import time

import httpx
import streamlit as st
from PIL import Image, ImageDraw

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


def _render_process_controls(annotated_images: list[dict]) -> None:
    """Render the process button and progress indicator."""
    st.subheader("Processing Controls")

    col1, col2 = st.columns([1, 3])

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


def _render_result_viewer(processed_images: list[dict]) -> None:
    """Render the result viewer with image and mask overlay."""
    if not processed_images:
        st.info("No processed images yet.")
        return

    # Get selected processed image
    if "selected_processed_index" not in st.session_state:
        st.session_state.selected_processed_index = 0

    if st.session_state.selected_processed_index >= len(processed_images):
        st.session_state.selected_processed_index = 0

    current_image = processed_images[st.session_state.selected_processed_index]

    # Header with filename and download button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(current_image["filename"])
    with col2:
        coco_data = _download_coco_json(current_image["id"])
        if coco_data:
            import json

            st.download_button(
                "Download COCO JSON",
                data=json.dumps(coco_data, indent=2),
                file_name=f"{current_image['filename'].rsplit('.', 1)[0]}_coco.json",
                mime="application/json",
            )

    # Display the image with overlays
    image_data = _fetch_image_data(current_image["id"])
    if image_data:
        mask_data = _fetch_mask_data(current_image["id"])
        annotations = _fetch_annotations(current_image["id"])
        overlay_image = _create_overlay_image(image_data, mask_data, annotations)
        st.image(overlay_image, use_container_width=True)
    else:
        st.error("Failed to load image")


def _render_navigation_controls(processed_images: list[dict]) -> None:
    """Render previous/next navigation buttons."""
    if len(processed_images) <= 1:
        return

    st.divider()
    nav_cols = st.columns([1, 3, 1])

    with nav_cols[0]:
        if st.button(
            "Previous", disabled=st.session_state.selected_processed_index == 0
        ):
            st.session_state.selected_processed_index -= 1
            st.rerun()

    with nav_cols[1]:
        st.caption(
            f"Image {st.session_state.selected_processed_index + 1} of {len(processed_images)}"
        )

    with nav_cols[2]:
        if st.button(
            "Next",
            disabled=st.session_state.selected_processed_index >= len(processed_images) - 1,
        ):
            st.session_state.selected_processed_index += 1
            st.rerun()


def _render_processed_gallery(processed_images: list[dict]) -> None:
    """Render the gallery of processed images."""
    st.subheader(f"Processed Images ({len(processed_images)})")

    if not processed_images:
        return

    num_cols = min(len(processed_images), 4)
    cols = st.columns(num_cols)

    for idx, img in enumerate(processed_images[:12]):
        with cols[idx % num_cols]:
            is_selected = idx == st.session_state.get("selected_processed_index", 0)
            if st.button(
                "Selected" if is_selected else "Select",
                key=f"proc_{img['id']}",
                disabled=is_selected,
            ):
                st.session_state.selected_processed_index = idx
                st.rerun()

            thumb_data = _fetch_image_data(img["id"])
            if thumb_data:
                st.image(thumb_data, use_container_width=True)
            st.caption(img["filename"])


def _get_status_color(status: str) -> str:
    """Get color for status badge."""
    colors = {
        "pending": "#3d3d4d",
        "annotated": "#1e40af",
        "processing": "#b45309",
        "processed": "#166534",
    }
    return colors.get(status, "#3d3d4d")


def render() -> None:
    """Render the processing page."""
    st.header("Process Images")
    st.caption("Run SAM3 inference on annotated images")

    # Initialize session state
    if "selected_processed_index" not in st.session_state:
        st.session_state.selected_processed_index = 0

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
    _render_process_controls(annotated_images)

    st.divider()

    # Layout: Result viewer on left, gallery on right
    viewer_col, gallery_col = st.columns([1, 1])

    with viewer_col:
        _render_result_viewer(processed_images)
        _render_navigation_controls(processed_images)

    with gallery_col:
        _render_processed_gallery(processed_images)

        # Show queued images
        if annotated_images:
            st.divider()
            st.subheader(f"Queued ({len(annotated_images)})")
            for img in annotated_images[:6]:
                status_color = _get_status_color(img.get("processing_status", "pending"))
                st.markdown(
                    f"<span style='color: {status_color};'>{img['filename']}</span>",
                    unsafe_allow_html=True,
                )
