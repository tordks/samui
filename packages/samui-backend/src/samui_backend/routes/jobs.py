"""Job management routes for processing jobs and queue."""

import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import desc
from sqlalchemy.orm import Session

from samui_backend.db.database import get_db
from samui_backend.db.models import Image, ProcessingJob, ProcessingResult
from samui_backend.dependencies import get_storage_service
from samui_backend.enums import JobStatus
from samui_backend.schemas import (
    ProcessingJobCreate,
    ProcessingJobResponse,
)
from samui_backend.services import (
    get_images_needing_processing,
    start_job_if_none_running,
)
from samui_backend.services.storage import StorageService

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("", response_model=ProcessingJobResponse, status_code=201)
def create_job(
    request: ProcessingJobCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> ProcessingJob:
    """Create a new processing job.

    For "Process" button: filters by needs_processing (force_all=False).
    For "Process All" button: uses all provided image_ids (force_all=True).

    Args:
        request: ProcessingJobCreate containing image_ids, mode, and force_all flag.
        background_tasks: FastAPI background tasks.
        db: Database session.

    Returns:
        ProcessingJobResponse with job details.

    Raises:
        HTTPException: If no valid images to process or image_ids don't exist.
    """
    # Validate image_ids exist and collect filenames
    valid_images: list[tuple[uuid.UUID, str]] = []
    for image_id in request.image_ids:
        image = db.get(Image, image_id)
        if image:
            valid_images.append((image_id, image.filename))

    if not valid_images:
        raise HTTPException(status_code=400, detail="No valid image IDs provided")

    valid_image_ids = [img_id for img_id, _ in valid_images]

    # Filter by needs_processing if not force_all
    if request.force_all:
        image_ids_to_process = valid_image_ids
    else:
        image_ids_to_process = get_images_needing_processing(db, valid_image_ids, request.mode)

    if not image_ids_to_process:
        raise HTTPException(status_code=400, detail="No images need processing")

    # Build filenames list for images to process (preserve order)
    id_to_filename = dict(valid_images)
    filenames_to_process = [id_to_filename[img_id] for img_id in image_ids_to_process]

    # Create job
    job = ProcessingJob(
        mode=request.mode,
        status=JobStatus.QUEUED,
        image_ids=[str(img_id) for img_id in image_ids_to_process],
        image_filenames=filenames_to_process,
        current_index=0,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Start job if none running
    start_job_if_none_running(db, background_tasks, job.id)

    return job


@router.get("", response_model=list[ProcessingJobResponse])
def list_jobs(db: Session = Depends(get_db)) -> list[ProcessingJob]:
    """List all processing jobs, newest first.

    Returns:
        List of ProcessingJobResponse objects.
    """
    jobs = db.query(ProcessingJob).order_by(desc(ProcessingJob.created_at)).all()
    return jobs


@router.get("/{job_id}", response_model=ProcessingJobResponse)
def get_job(job_id: uuid.UUID, db: Session = Depends(get_db)) -> ProcessingJob:
    """Get job details and progress.

    Args:
        job_id: UUID of the job.
        db: Database session.

    Returns:
        ProcessingJobResponse with current progress.

    Raises:
        HTTPException: If job not found.
    """
    job = db.get(ProcessingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


results_router = APIRouter(prefix="/results", tags=["results"])


@results_router.get("/{result_id}/mask")
def get_result_mask(
    result_id: uuid.UUID,
    db: Session = Depends(get_db),
    storage: StorageService = Depends(get_storage_service),
) -> Response:
    """Get mask PNG for a specific processing result.

    Args:
        result_id: UUID of the processing result.
        db: Database session.
        storage: Storage service.

    Returns:
        PNG mask image.

    Raises:
        HTTPException: If result not found or mask unavailable.
    """
    result = db.get(ProcessingResult, result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Processing result not found")

    if not result.mask_blob_path:
        raise HTTPException(status_code=404, detail="Mask not available for this result")

    try:
        mask_bytes = storage.get_image(result.mask_blob_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve mask: {e}") from e

    return Response(content=mask_bytes, media_type="image/png")
