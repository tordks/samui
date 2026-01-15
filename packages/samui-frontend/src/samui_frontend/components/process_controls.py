"""Processing controls component for triggering SAM3 inference."""

import streamlit as st

from samui_frontend.api import create_job, fetch_annotations, fetch_job
from samui_frontend.models import SegmentationMode


def get_images_ready_for_mode(images: list[dict], mode: SegmentationMode) -> list[dict]:
    """Filter images that are ready for processing in the given mode.

    For inside_box: images with SEGMENT annotations
    For find_all: images with text_prompt OR exemplar annotations
    """
    ready = []
    for img in images:
        if mode == SegmentationMode.INSIDE_BOX:
            annotations = fetch_annotations(img["id"], mode)
            if annotations:
                ready.append(img)
        else:
            has_text = bool(img.get("text_prompt"))
            annotations = fetch_annotations(img["id"], mode)
            if has_text or annotations:
                ready.append(img)
    return ready


@st.fragment(run_every=1)
def render_processing_status() -> None:
    """Fragment that polls processing status with auto-refresh.

    Shows a simple indicator: running spinner or idle status.
    Detailed job status is available on the Jobs page.
    """
    job_id = st.session_state.get("current_job_id")

    if job_id:
        job = fetch_job(job_id)
        if job and job.get("is_running"):
            st.status("Processing...", state="running")
        elif job and job.get("error"):
            st.error("Job failed")
            st.session_state.current_job_id = None
        elif job and job.get("status") == "completed":
            st.success("Done")
            st.session_state.current_job_id = None
        else:
            st.session_state.current_job_id = None


def render_process_buttons(ready_images: list[dict], mode: SegmentationMode) -> None:
    """Render Process and Process All buttons."""
    col1, col2 = st.columns(2)

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
                st.rerun()
            else:
                st.error("Failed to create processing job")
