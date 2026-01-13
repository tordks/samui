"""Jobs page showing all processing jobs with status."""

from datetime import datetime

import streamlit as st

from samui_frontend.api import fetch_jobs


def _format_timestamp(timestamp_str: str | None) -> str:
    """Format ISO timestamp to human-readable string."""
    if not timestamp_str:
        return "-"
    try:
        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError):
        return timestamp_str


def _calculate_duration(started_at: str | None, completed_at: str | None) -> str:
    """Calculate and format duration between two timestamps."""
    if not started_at or not completed_at:
        return "-"
    try:
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        end = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
        delta = end - start
        total_seconds = int(delta.total_seconds())

        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes}m {seconds}s"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours}h {minutes}m"
    except (ValueError, AttributeError):
        return "-"


def _get_status_icon(status: str) -> str:
    """Return an icon/emoji for the job status."""
    icons = {
        "queued": "⏳",
        "running": "🔄",
        "completed": "✅",
        "failed": "❌",
    }
    return icons.get(status, "❓")


def _get_mode_display(mode: str) -> str:
    """Return human-readable mode name."""
    modes = {
        "inside_box": "Inside Box",
        "find_all": "Find All",
    }
    return modes.get(mode, mode)


def _render_job_line(job: dict) -> None:
    """Render a single job as a compact one-line entry."""
    status = job.get("status", "unknown")
    mode = job.get("mode", "unknown")
    image_count = job.get("image_count", 0)
    is_running = job.get("is_running", False)
    processed_count = job.get("processed_count", 0)
    current_image = job.get("current_image_filename")
    started_at = job.get("started_at")
    completed_at = job.get("completed_at")

    status_icon = _get_status_icon(status)
    mode_display = _get_mode_display(mode)

    # Build the line parts
    parts = [f"{status_icon} **{mode_display}**"]

    if is_running:
        parts.append(f"{processed_count}/{image_count} images")
        if current_image:
            parts.append(f"*{current_image}*")
    else:
        parts.append(f"{image_count} images")

    parts.append(_format_timestamp(job.get("created_at")))

    if status == "completed":
        parts.append(_calculate_duration(started_at, completed_at))

    st.markdown(" | ".join(parts))

    # Show error on second line for failed jobs
    if status == "failed" and job.get("error"):
        st.caption(f"Error: {job.get('error')}")


def render() -> None:
    """Render the Jobs page."""
    st.header("Processing Jobs")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("View all processing jobs and their status")
    with col2:
        if st.button("Refresh"):
            st.rerun()

    # Fetch jobs
    jobs = fetch_jobs()

    if not jobs:
        st.info("No processing jobs yet. Start processing from the Processing page.")
        return

    # Summary stats
    queued = sum(1 for j in jobs if j.get("status") == "queued")
    running = sum(1 for j in jobs if j.get("status") == "running")
    completed = sum(1 for j in jobs if j.get("status") == "completed")
    failed = sum(1 for j in jobs if j.get("status") == "failed")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Queued", queued)
    with col2:
        st.metric("Running", running)
    with col3:
        st.metric("Completed", completed)
    with col4:
        st.metric("Failed", failed)

    st.divider()

    # Render each job
    for job in jobs:
        _render_job_line(job)
