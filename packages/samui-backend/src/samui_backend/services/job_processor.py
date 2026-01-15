"""Job processor service for managing processing jobs and queue."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage
from sqlalchemy import desc
from sqlalchemy.orm import Session

from samui_backend.db.database import get_background_db
from samui_backend.db.models import BboxAnnotation, Image, PointAnnotation, ProcessingJob, ProcessingResult
from samui_backend.enums import JobStatus, PromptType, SegmentationMode
from samui_backend.services.coco_export import generate_coco_json
from samui_backend.services.sam3_inference import SAM3Service
from samui_backend.services.storage import StorageService

if TYPE_CHECKING:
    from fastapi import BackgroundTasks

logger = logging.getLogger(__name__)


def get_latest_result(db: Session, image_id: uuid.UUID, mode: SegmentationMode) -> ProcessingResult | None:
    """Get the most recent processing result for an image and mode."""
    return (
        db.query(ProcessingResult)
        .filter(ProcessingResult.image_id == image_id, ProcessingResult.mode == mode)
        .order_by(desc(ProcessingResult.processed_at))
        .first()
    )


def get_annotations_for_mode(db: Session, image_id: uuid.UUID, mode: SegmentationMode) -> list[BboxAnnotation]:
    """Get bbox annotations relevant to the given segmentation mode."""
    if mode == SegmentationMode.INSIDE_BOX:
        return (
            db.query(BboxAnnotation)
            .filter(BboxAnnotation.image_id == image_id, BboxAnnotation.prompt_type == PromptType.SEGMENT)
            .all()
        )
    elif mode == SegmentationMode.FIND_ALL:
        # Find-all mode uses exemplar annotations
        return (
            db.query(BboxAnnotation)
            .filter(
                BboxAnnotation.image_id == image_id,
                BboxAnnotation.prompt_type.in_([PromptType.POSITIVE_EXEMPLAR, PromptType.NEGATIVE_EXEMPLAR]),
            )
            .all()
        )
    else:
        # POINT mode doesn't use BboxAnnotation
        return []


def get_point_annotations_for_image(db: Session, image_id: uuid.UUID) -> list[PointAnnotation]:
    """Get point annotations for an image (used in POINT mode)."""
    return db.query(PointAnnotation).filter(PointAnnotation.image_id == image_id).all()


def needs_processing(db: Session, image_id: uuid.UUID, mode: SegmentationMode) -> bool:
    """Derive processing need by comparing current annotations vs last result.

    Returns True if:
    - No previous result exists and there's something to process
    - Annotation set has changed (additions or deletions)
    - Text prompt has changed (find-all mode only)
    """
    image = db.get(Image, image_id)
    if not image:
        return False

    # Get annotation IDs based on mode
    if mode == SegmentationMode.POINT:
        point_annotations = get_point_annotations_for_image(db, image_id)
        current_ids = {str(a.id) for a in point_annotations}
    else:
        annotations = get_annotations_for_mode(db, image_id, mode)
        current_ids = {str(a.id) for a in annotations}

    latest_result = get_latest_result(db, image_id, mode)

    # No previous result - process if there's something to process
    if not latest_result:
        has_annotations = bool(current_ids)
        has_find_all_prompt = mode == SegmentationMode.FIND_ALL and bool(image.text_prompt)
        return has_annotations or has_find_all_prompt

    # Compare annotation sets (catches adds AND deletes)
    processed_ids = set(latest_result.annotation_ids or [])
    if current_ids != processed_ids:
        return True

    # Check text_prompt changed (find-all mode)
    return mode == SegmentationMode.FIND_ALL and image.text_prompt != latest_result.text_prompt_used


def get_images_needing_processing(db: Session, image_ids: list[uuid.UUID], mode: SegmentationMode) -> list[uuid.UUID]:
    """Filter image_ids to those that need processing."""
    return [img_id for img_id in image_ids if needs_processing(db, img_id, mode)]


def _save_mask_to_storage(
    storage: StorageService,
    masks: NDArray[np.uint8],
    result_id: uuid.UUID,
) -> str:
    """Save combined mask image to storage using result_id for history support."""
    mask_blob_path = f"masks/{result_id}.png"

    if masks.size == 0:
        # Empty mask for no detections
        combined_mask = np.zeros((1, 1), dtype=np.uint8)
    else:
        combined_mask = np.zeros((masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for mask in masks:
            combined_mask = np.maximum(combined_mask, mask)

    mask_image = PILImage.fromarray(combined_mask)
    mask_buffer = BytesIO()
    mask_image.save(mask_buffer, format="PNG")
    mask_bytes = mask_buffer.getvalue()

    storage.upload_blob(mask_blob_path, mask_bytes, content_type="image/png")
    return mask_blob_path


def _save_coco_to_storage(
    storage: StorageService,
    image: Image,
    bboxes: list[tuple[int, int, int, int]],
    masks: NDArray[np.uint8],
    result_id: uuid.UUID,
    points: list[tuple[int, int, bool]] | None = None,
) -> str:
    """Generate and save COCO JSON to storage using result_id for history support."""
    coco_blob_path = f"coco/{result_id}.json"
    coco_json = generate_coco_json(
        image_id=image.id,
        filename=image.filename,
        width=image.width,
        height=image.height,
        bboxes=bboxes,
        masks=masks,
        points=points,
    )
    coco_bytes = json.dumps(coco_json, indent=2).encode("utf-8")
    storage.upload_blob(coco_blob_path, coco_bytes, content_type="application/json")
    return coco_blob_path


def _process_inside_box(
    storage: StorageService,
    sam3: SAM3Service,
    image: Image,
    pil_image: PILImage.Image,
    annotations: list[BboxAnnotation],
    result: ProcessingResult,
) -> bool:
    """Process image with bounding box prompts (INSIDE_BOX mode).

    Returns:
        True if successful, False if no annotations to process.
    """
    if not annotations:
        logger.warning(f"No segment annotations for image {image.id}, skipping")
        return False

    bboxes = [(ann.bbox_x, ann.bbox_y, ann.bbox_width, ann.bbox_height) for ann in annotations]
    masks = sam3.process_image(pil_image, bboxes)

    result.mask_blob_path = _save_mask_to_storage(storage, masks, result.id)
    result.coco_json_blob_path = _save_coco_to_storage(storage, image, bboxes, masks, result.id)
    return True


def _process_find_all(
    storage: StorageService,
    sam3: SAM3Service,
    image: Image,
    pil_image: PILImage.Image,
    annotations: list[BboxAnnotation],
    result: ProcessingResult,
) -> bool:
    """Process image with text prompt and/or exemplar boxes (FIND_ALL mode).

    Returns:
        True if successful, False if no text prompt or exemplars to process.
    """
    text_prompt = image.text_prompt

    if not text_prompt and not annotations:
        logger.warning(f"No text prompt or exemplars for image {image.id} in find-all mode, skipping")
        return False

    # Convert exemplar annotations to (bbox_xywh, is_positive) format
    exemplar_boxes = None
    if annotations:
        exemplar_boxes = [
            (
                (ann.bbox_x, ann.bbox_y, ann.bbox_width, ann.bbox_height),
                ann.prompt_type == PromptType.POSITIVE_EXEMPLAR,
            )
            for ann in annotations
        ]

    # Run find-all inference
    find_result = sam3.process_image_find_all(pil_image, text_prompt, exemplar_boxes)

    # Store discovered bboxes in result
    result.bboxes = [{"x": x, "y": y, "width": w, "height": h} for x, y, w, h in find_result.bboxes]

    # Save masks and COCO JSON
    if find_result.masks.size > 0:
        result.mask_blob_path = _save_mask_to_storage(storage, find_result.masks, result.id)
        result.coco_json_blob_path = _save_coco_to_storage(
            storage, image, find_result.bboxes, find_result.masks, result.id
        )
    else:
        # No discoveries - save empty results
        empty_masks = np.zeros((0, pil_image.height, pil_image.width), dtype=np.uint8)
        result.mask_blob_path = _save_mask_to_storage(storage, empty_masks, result.id)
        result.coco_json_blob_path = _save_coco_to_storage(storage, image, [], empty_masks, result.id)

    logger.info(f"Find-all discovered {len(find_result.bboxes)} objects for image {image.id}")
    return True


def _process_point(
    storage: StorageService,
    sam3: SAM3Service,
    image: Image,
    pil_image: PILImage.Image,
    point_annotations: list[PointAnnotation],
    result: ProcessingResult,
) -> bool:
    """Process image with point prompts (POINT mode).

    Returns:
        True if successful, False if no point annotations to process.
    """
    if not point_annotations:
        logger.warning(f"No point annotations for image {image.id}, skipping")
        return False

    # Extract coordinates and labels from point annotations
    points = [(ann.point_x, ann.point_y) for ann in point_annotations]
    labels = [1 if ann.is_positive else 0 for ann in point_annotations]

    # Run point-based inference
    masks = sam3.process_image_points(pil_image, points, labels)

    # Prepare points metadata for COCO export (x, y, is_positive)
    points_metadata = [(ann.point_x, ann.point_y, ann.is_positive) for ann in point_annotations]

    # Save mask (bboxes computed from mask, points stored in metadata)
    result.mask_blob_path = _save_mask_to_storage(storage, masks, result.id)
    result.coco_json_blob_path = _save_coco_to_storage(storage, image, [], masks, result.id, points=points_metadata)

    logger.info(f"Point mode processed {len(points)} points for image {image.id}")
    return True


def process_single_image(
    db: Session,
    storage: StorageService,
    sam3: SAM3Service,
    image: Image,
    job: ProcessingJob,
    mode: SegmentationMode,
) -> ProcessingResult | None:
    """Process a single image through SAM3 inference.

    Dispatches to mode-specific helper functions for the actual processing.

    Returns:
        ProcessingResult if successful, None otherwise.
    """
    # Get annotations based on mode
    if mode == SegmentationMode.POINT:
        point_annotations = get_point_annotations_for_image(db, image.id)
        annotation_ids = [str(a.id) for a in point_annotations]
        bbox_annotations: list[BboxAnnotation] = []
    else:
        bbox_annotations = get_annotations_for_mode(db, image.id, mode)
        annotation_ids = [str(a.id) for a in bbox_annotations]
        point_annotations: list[PointAnnotation] = []

    # Load image data
    image_data = storage.get_image(image.blob_path)
    pil_image = PILImage.open(BytesIO(image_data)).convert("RGB")

    # Create result first to get ID for blob paths
    result = ProcessingResult(
        job_id=job.id,
        image_id=image.id,
        mode=mode,
        annotation_ids=annotation_ids,
        text_prompt_used=image.text_prompt if mode == SegmentationMode.FIND_ALL else None,
        bboxes=None,
        mask_blob_path="",  # Will be updated by helper
        coco_json_blob_path="",  # Will be updated by helper
    )
    db.add(result)
    db.flush()  # Get the result.id

    # Dispatch to mode-specific processing
    if mode == SegmentationMode.INSIDE_BOX:
        success = _process_inside_box(storage, sam3, image, pil_image, bbox_annotations, result)
    elif mode == SegmentationMode.FIND_ALL:
        success = _process_find_all(storage, sam3, image, pil_image, bbox_annotations, result)
    elif mode == SegmentationMode.POINT:
        success = _process_point(storage, sam3, image, pil_image, point_annotations, result)
    else:
        logger.error(f"Unknown segmentation mode: {mode}")
        db.rollback()
        return None

    if not success:
        db.rollback()
        return None

    db.commit()
    logger.info(f"Processed image {image.id} ({image.filename}) with mode {mode.value}")
    return result


def process_job(job_id: uuid.UUID) -> None:
    """Process a job, updating status and results in the database.

    Sets job status to RUNNING, processes each image, then sets to COMPLETED or FAILED.
    """
    # Import here to avoid circular import
    from samui_backend.dependencies import get_sam3_service, get_storage_service

    with get_background_db() as db:
        job = db.get(ProcessingJob, job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return

        # Mark job as running
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(UTC)
        db.commit()

        storage = get_storage_service()
        sam3 = get_sam3_service()

        try:
            sam3.load_model()

            image_ids = [uuid.UUID(img_id) for img_id in job.image_ids]
            mode = job.mode

            for idx, image_id in enumerate(image_ids):
                job.current_index = idx
                db.commit()

                image = db.get(Image, image_id)
                if not image:
                    logger.warning(f"Image {image_id} not found, skipping")
                    continue

                try:
                    process_single_image(db, storage, sam3, image, job, mode)
                except Exception as e:
                    logger.error(f"Error processing image {image_id}: {e}")
                    db.rollback()

            # Mark job as completed
            job.current_index = len(image_ids)
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now(UTC)
            db.commit()

        except Exception as e:
            logger.exception(f"Job {job_id} failed: {e}")
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now(UTC)
            db.commit()
        finally:
            sam3.unload_model()


def process_job_and_check_queue(job_id: uuid.UUID) -> None:
    """Process a job, then check for and process any queued jobs."""
    process_job(job_id)

    # Check for next queued job
    with get_background_db() as db:
        next_job = (
            db.query(ProcessingJob)
            .filter(ProcessingJob.status == JobStatus.QUEUED)
            .order_by(ProcessingJob.created_at)
            .first()
        )

        if next_job:
            logger.info(f"Found queued job {next_job.id}, processing...")
            process_job_and_check_queue(next_job.id)


def start_job_if_none_running(db: Session, background_tasks: BackgroundTasks, job_id: uuid.UUID) -> bool:
    """Start job processing if no job is currently running.

    Args:
        db: Database session.
        background_tasks: FastAPI BackgroundTasks.
        job_id: ID of the job to potentially start.

    Returns:
        True if job was started, False if another job is running.
    """
    # Check if any job is currently running
    running_job = db.query(ProcessingJob).filter(ProcessingJob.status == JobStatus.RUNNING).first()

    if running_job:
        logger.info(f"Job {running_job.id} is already running, {job_id} will stay queued")
        return False

    # Start background task for this job
    background_tasks.add_task(process_job_and_check_queue, job_id)
    logger.info(f"Started background task for job {job_id}")
    return True


def cleanup_stale_jobs(db: Session) -> int:
    """Reset any jobs in RUNNING state to FAILED.

    Called on application startup to handle jobs that were interrupted
    by server crash/restart.

    Returns:
        Number of jobs reset.
    """
    stale_jobs = db.query(ProcessingJob).filter(ProcessingJob.status == JobStatus.RUNNING).all()

    count = 0
    for job in stale_jobs:
        job.status = JobStatus.FAILED
        job.error = "Server restarted while job was running"
        job.completed_at = datetime.now(UTC)
        count += 1
        logger.warning(f"Reset stale job {job.id} to FAILED")

    if count > 0:
        db.commit()

    return count
