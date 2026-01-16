"""API client functions for backend communication."""

import httpx

from samui_frontend.config import API_URL
from samui_frontend.constants import API_TIMEOUT_READ, API_TIMEOUT_WRITE
from samui_frontend.models import PromptType, SegmentationMode


def fetch_images() -> list[dict]:
    """Fetch all images from the API."""
    try:
        response = httpx.get(f"{API_URL}/images", timeout=API_TIMEOUT_READ)
        response.raise_for_status()
        return response.json().get("images", [])
    except httpx.HTTPError:
        return []


def fetch_image_data(image_id: str) -> bytes | None:
    """Fetch image data from the API."""
    try:
        response = httpx.get(f"{API_URL}/images/{image_id}/data", timeout=API_TIMEOUT_READ)
        if response.status_code == 200:
            return response.content
    except httpx.HTTPError:
        pass
    return None


def fetch_annotations(
    image_id: str,
    mode: SegmentationMode | None = None,
) -> list[dict]:
    """Fetch annotations for an image, optionally filtered by segmentation mode.

    Args:
        image_id: The image UUID.
        mode: If provided, filter annotations by mode:
            - INSIDE_BOX: only SEGMENT prompt_type
            - FIND_ALL: both POSITIVE_EXEMPLAR and NEGATIVE_EXEMPLAR prompt_types
    """
    try:
        if mode is None:
            response = httpx.get(f"{API_URL}/annotations/{image_id}", timeout=API_TIMEOUT_READ)
            response.raise_for_status()
            return response.json().get("annotations", [])

        if mode == SegmentationMode.INSIDE_BOX:
            response = httpx.get(
                f"{API_URL}/annotations/{image_id}",
                params={"prompt_type": PromptType.SEGMENT.value},
                timeout=API_TIMEOUT_READ,
            )
            response.raise_for_status()
            return response.json().get("annotations", [])

        # FIND_ALL mode - fetch user-provided exemplars (positive and negative)
        annotations = []
        for pt in [PromptType.POSITIVE_EXEMPLAR, PromptType.NEGATIVE_EXEMPLAR]:
            response = httpx.get(
                f"{API_URL}/annotations/{image_id}",
                params={"prompt_type": pt.value},
                timeout=API_TIMEOUT_READ,
            )
            response.raise_for_status()
            annotations.extend(response.json().get("annotations", []))
        return annotations

    except httpx.HTTPError:
        return []


def create_annotation(
    image_id: str,
    x: int,
    y: int,
    width: int,
    height: int,
    prompt_type: PromptType = PromptType.SEGMENT,
) -> bool:
    """Create a new annotation with the specified prompt type."""
    try:
        response = httpx.post(
            f"{API_URL}/annotations",
            json={
                "image_id": image_id,
                "bbox_x": x,
                "bbox_y": y,
                "bbox_width": width,
                "bbox_height": height,
                "prompt_type": prompt_type.value,
            },
            timeout=API_TIMEOUT_READ,
        )
        response.raise_for_status()
        return True
    except httpx.HTTPError:
        return False


def delete_annotation(annotation_id: str) -> bool:
    """Delete an annotation."""
    try:
        response = httpx.delete(f"{API_URL}/annotations/{annotation_id}", timeout=API_TIMEOUT_READ)
        response.raise_for_status()
        return True
    except httpx.HTTPError:
        return False


def update_image_text_prompt(image_id: str, text_prompt: str | None) -> bool:
    """Update or clear the text prompt for an image.

    Args:
        image_id: The image UUID.
        text_prompt: The text prompt to set, or None to clear.
    """
    try:
        response = httpx.patch(
            f"{API_URL}/images/{image_id}",
            json={"text_prompt": text_prompt},
            timeout=API_TIMEOUT_READ,
        )
        response.raise_for_status()
        return True
    except httpx.HTTPError:
        return False


def delete_image(image_id: str) -> bool:
    """Delete an image."""
    try:
        response = httpx.delete(f"{API_URL}/images/{image_id}", timeout=API_TIMEOUT_READ)
        response.raise_for_status()
        return True
    except httpx.HTTPError:
        return False


def upload_image(filename: str, content: bytes, content_type: str) -> dict | None:
    """Upload an image to the API.

    Returns the image metadata dict on success, None on failure.
    """
    try:
        files = {"file": (filename, content, content_type)}
        response = httpx.post(f"{API_URL}/images", files=files, timeout=API_TIMEOUT_WRITE)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


def download_coco_json(image_id: str, mode: SegmentationMode | None = None) -> dict | None:
    """Download COCO JSON for an image."""
    try:
        params = {}
        if mode:
            params["mode"] = mode.value
        response = httpx.get(
            f"{API_URL}/process/export/{image_id}",
            params=params if params else None,
            timeout=API_TIMEOUT_WRITE,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


def create_job(image_ids: list[str], mode: SegmentationMode, force_all: bool = False) -> dict | None:
    """Create a new processing job.

    Args:
        image_ids: List of image UUIDs to process.
        mode: The segmentation mode to use.
        force_all: If True, process all images. If False, only process changed images.

    Returns:
        The job response dict on success, dict with "error" key on API error, None on network error.
    """
    try:
        response = httpx.post(
            f"{API_URL}/jobs",
            json={
                "image_ids": image_ids,
                "mode": mode.value,
                "force_all": force_all,
            },
            timeout=API_TIMEOUT_WRITE,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        # Extract error detail from API response
        try:
            detail = e.response.json().get("detail", "Unknown error")
        except Exception:
            detail = str(e)
        return {"error": detail}
    except httpx.HTTPError:
        return None


def fetch_jobs() -> list[dict]:
    """Fetch all processing jobs.

    Returns:
        List of job dicts, newest first.
    """
    try:
        response = httpx.get(f"{API_URL}/jobs", timeout=API_TIMEOUT_READ)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return []


def fetch_job(job_id: str) -> dict | None:
    """Fetch a single job's details including progress status.

    Args:
        job_id: The job UUID.

    Returns:
        Job dict with status fields (is_running, processed_count, current_image_filename),
        or None on failure.
    """
    try:
        response = httpx.get(f"{API_URL}/jobs/{job_id}", timeout=API_TIMEOUT_READ)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


def fetch_image_history(image_id: str, mode: SegmentationMode) -> list[dict]:
    """Fetch processing history for an image.

    Args:
        image_id: The image UUID.
        mode: The segmentation mode to filter by.

    Returns:
        List of processing result dicts, newest first.
    """
    try:
        response = httpx.get(
            f"{API_URL}/images/{image_id}/history",
            params={"mode": mode.value},
            timeout=API_TIMEOUT_READ,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return []


def fetch_all_history(mode: SegmentationMode) -> list[dict]:
    """Fetch all processing history across all images.

    Args:
        mode: The segmentation mode to filter by.

    Returns:
        List of processing result dicts, newest first.
    """
    try:
        response = httpx.get(
            f"{API_URL}/images/history",
            params={"mode": mode.value},
            timeout=API_TIMEOUT_READ,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return []


def fetch_result_mask(result_id: str) -> bytes | None:
    """Fetch mask PNG for a specific processing result.

    Args:
        result_id: The processing result UUID.

    Returns:
        Mask PNG bytes on success, None on failure.
    """
    try:
        response = httpx.get(
            f"{API_URL}/results/{result_id}/mask",
            timeout=API_TIMEOUT_READ,
        )
        if response.status_code == 200:
            return response.content
    except httpx.HTTPError:
        pass
    return None


def create_point_annotation(image_id: str, x: int, y: int, is_positive: bool) -> dict | None:
    """Create a point annotation for an image.

    Args:
        image_id: The image UUID.
        x: X coordinate of the point.
        y: Y coordinate of the point.
        is_positive: True for positive (foreground), False for negative (background).

    Returns:
        The created annotation dict on success, None on failure.
    """
    try:
        response = httpx.post(
            f"{API_URL}/point-annotations",
            json={
                "image_id": image_id,
                "point_x": x,
                "point_y": y,
                "is_positive": is_positive,
            },
            timeout=API_TIMEOUT_READ,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


def fetch_point_annotations(image_id: str) -> list[dict]:
    """Fetch all point annotations for an image.

    Args:
        image_id: The image UUID.

    Returns:
        List of point annotation dicts.
    """
    try:
        response = httpx.get(
            f"{API_URL}/point-annotations/{image_id}",
            timeout=API_TIMEOUT_READ,
        )
        response.raise_for_status()
        return response.json().get("annotations", [])
    except httpx.HTTPError:
        return []


def delete_point_annotation(annotation_id: str) -> bool:
    """Delete a point annotation.

    Args:
        annotation_id: The point annotation UUID.

    Returns:
        True on success, False on failure.
    """
    try:
        response = httpx.delete(
            f"{API_URL}/point-annotations/{annotation_id}",
            timeout=API_TIMEOUT_READ,
        )
        response.raise_for_status()
        return True
    except httpx.HTTPError:
        return False
