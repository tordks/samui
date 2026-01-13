"""Processing routes for SAM3 inference and COCO export.

Note: Job creation and management is now handled by routes/jobs.py.
This module provides legacy-compatible endpoints for mask/export retrieval.
"""

import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, Response
from sqlalchemy import desc
from sqlalchemy.orm import Session

from samui_backend.db.database import get_db
from samui_backend.db.models import Image, ProcessingJob, ProcessingResult
from samui_backend.dependencies import get_storage_service
from samui_backend.enums import JobStatus, SegmentationMode
from samui_backend.schemas import ProcessStatus
from samui_backend.services.storage import StorageService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/process", tags=["processing"])


@router.get("/status", response_model=ProcessStatus)
def get_processing_status(db: Session = Depends(get_db)) -> ProcessStatus:
    """Get current processing status by querying the job table.

    Returns:
        ProcessStatus derived from the current RUNNING job,
        or the most recent COMPLETED/FAILED job if none is running.
    """
    # Get RUNNING job first, then fall back to latest completed/failed
    job = (
        db.query(ProcessingJob)
        .filter(ProcessingJob.status == JobStatus.RUNNING)
        .order_by(desc(ProcessingJob.started_at))
        .first()
    )
    if not job:
        job = (
            db.query(ProcessingJob)
            .filter(ProcessingJob.status.in_([JobStatus.COMPLETED, JobStatus.FAILED]))
            .order_by(desc(ProcessingJob.completed_at), desc(ProcessingJob.started_at))
            .first()
        )

    if not job:
        return ProcessStatus(
            batch_id=None,
            is_running=False,
            processed_count=0,
            total_count=0,
            current_image_id=None,
            current_image_filename=None,
            error=None,
        )

    is_running = job.status == JobStatus.RUNNING
    total = len(job.image_ids)
    processed = job.current_index if is_running else total

    # Get current image info for running jobs
    current_image_id = None
    current_image_filename = None
    if is_running and job.current_index < total:
        current_image_id = uuid.UUID(job.image_ids[job.current_index])
        current_image = db.get(Image, current_image_id)
        if current_image:
            current_image_filename = current_image.filename

    return ProcessStatus(
        batch_id=job.id,
        is_running=is_running,
        processed_count=processed,
        total_count=total,
        current_image_id=current_image_id,
        current_image_filename=current_image_filename,
        error=job.error,
    )


@router.get("/mask/{image_id}")
def get_mask(
    image_id: uuid.UUID,
    mode: SegmentationMode = SegmentationMode.INSIDE_BOX,
    db: Session = Depends(get_db),
    storage: StorageService = Depends(get_storage_service),
):
    """Get segmentation mask image for a processed image.

    Returns the latest processing result's mask for the given image and mode.

    Args:
        image_id: UUID of the image.
        mode: Segmentation mode (defaults to INSIDE_BOX for backward compatibility).
        db: Database session.
        storage: Storage service.

    Returns:
        PNG mask image.

    Raises:
        HTTPException: If image not found or not processed.
    """
    # Get the latest result for this image and mode
    result = (
        db.query(ProcessingResult)
        .filter(ProcessingResult.image_id == image_id, ProcessingResult.mode == mode)
        .order_by(desc(ProcessingResult.processed_at))
        .first()
    )
    if not result or not result.mask_blob_path:
        raise HTTPException(status_code=404, detail=f"Mask not found for this image with mode {mode.value}")

    try:
        mask_bytes = storage.get_image(result.mask_blob_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve mask: {e}") from e

    return Response(content=mask_bytes, media_type="image/png")


@router.get("/export/{image_id}")
def export_coco_json(
    image_id: uuid.UUID,
    mode: SegmentationMode = SegmentationMode.INSIDE_BOX,
    db: Session = Depends(get_db),
    storage: StorageService = Depends(get_storage_service),
) -> Response:
    """Download COCO JSON annotation for a processed image.

    Returns the latest processing result's COCO JSON for the given image and mode.

    Args:
        image_id: UUID of the image to export.
        mode: Segmentation mode (defaults to INSIDE_BOX for backward compatibility).
        db: Database session.
        storage: Storage service.

    Returns:
        COCO JSON content as downloadable file.

    Raises:
        HTTPException: If image not found or not processed.
    """
    # Get image
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Get the latest result for this image and mode
    result = (
        db.query(ProcessingResult)
        .filter(ProcessingResult.image_id == image_id, ProcessingResult.mode == mode)
        .order_by(desc(ProcessingResult.processed_at))
        .first()
    )
    if not result:
        raise HTTPException(status_code=404, detail=f"Image has not been processed yet with mode {mode.value}")

    # Fetch COCO JSON from storage
    try:
        coco_bytes = storage.get_image(result.coco_json_blob_path)
        coco_json = json.loads(coco_bytes.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve COCO JSON: {e}") from e

    # Return as downloadable JSON with filename
    return JSONResponse(
        content=coco_json,
        headers={"Content-Disposition": f'attachment; filename="{image.filename.rsplit(".", 1)[0]}_coco.json"'},
    )
