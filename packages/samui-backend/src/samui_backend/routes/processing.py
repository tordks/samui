"""Processing routes for SAM3 inference and COCO export."""

import json
import logging
import uuid
from io import BytesIO

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from numpy.typing import NDArray
from PIL import Image as PILImage
from sqlalchemy.orm import Session

from samui_backend.db.database import get_db, SessionLocal
from samui_backend.db.models import Annotation, Image, ProcessingResult, ProcessingStatus
from samui_backend.schemas import ProcessRequest, ProcessResponse, ProcessStatus
from samui_backend.services import SAM3Service, StorageService, generate_coco_json

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/process", tags=["processing"])

# Singleton services
_storage_service: StorageService | None = None
_sam3_service: SAM3Service | None = None

# Processing state (in-memory for single-user tool)
_processing_state: dict = {
    "batch_id": None,
    "is_running": False,
    "processed_count": 0,
    "total_count": 0,
    "current_image_id": None,
    "current_image_filename": None,
    "error": None,
}


def get_storage_service() -> StorageService:
    """Get or create the storage service singleton."""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service


def get_sam3_service() -> SAM3Service:
    """Get or create the SAM3 service singleton."""
    global _sam3_service
    if _sam3_service is None:
        _sam3_service = SAM3Service()
    return _sam3_service


def _save_mask_to_storage(storage: StorageService, masks: NDArray[np.uint8], image_id: uuid.UUID) -> str:
    """Save combined mask image to storage."""
    mask_blob_path = f"masks/{image_id}.png"
    combined_mask = np.zeros((masks.shape[1], masks.shape[2]), dtype=np.uint8)
    for mask in masks:
        combined_mask = np.maximum(combined_mask, mask)

    mask_image = PILImage.fromarray(combined_mask)
    mask_buffer = BytesIO()
    mask_image.save(mask_buffer, format="PNG")
    mask_bytes = mask_buffer.getvalue()

    blob_client = storage._client.get_blob_client(container=storage._container_name, blob=mask_blob_path)
    blob_client.upload_blob(mask_bytes, overwrite=True)
    return mask_blob_path


def _save_coco_to_storage(
    storage: StorageService,
    image: Image,
    bboxes: list[tuple[int, int, int, int]],
    masks: NDArray[np.uint8],
) -> str:
    """Generate and save COCO JSON to storage."""
    coco_blob_path = f"coco/{image.id}.json"
    coco_json = generate_coco_json(
        image_id=image.id,
        filename=image.filename,
        width=image.width,
        height=image.height,
        bboxes=bboxes,
        masks=masks,
    )
    coco_bytes = json.dumps(coco_json, indent=2).encode("utf-8")
    blob_client = storage._client.get_blob_client(container=storage._container_name, blob=coco_blob_path)
    blob_client.upload_blob(coco_bytes, overwrite=True)
    return coco_blob_path


def _process_single_image(
    db: Session,
    storage: StorageService,
    sam3: SAM3Service,
    image: Image,
    batch_id: uuid.UUID,
    existing_result: ProcessingResult | None,
) -> bool:
    """Process a single image through SAM3 inference.

    Returns:
        True if processed successfully, False otherwise.
    """
    image_data = storage.get_image(image.blob_path)
    pil_image = PILImage.open(BytesIO(image_data)).convert("RGB")

    annotations = db.query(Annotation).filter(Annotation.image_id == image.id).all()
    if not annotations:
        logger.warning(f"No annotations for image {image.id}, skipping")
        image.processing_status = ProcessingStatus.ANNOTATED
        db.commit()
        return False

    bboxes = [(ann.bbox_x, ann.bbox_y, ann.bbox_width, ann.bbox_height) for ann in annotations]

    masks = sam3.process_image(pil_image, bboxes)
    mask_blob_path = _save_mask_to_storage(storage, masks, image.id)
    coco_blob_path = _save_coco_to_storage(storage, image, bboxes, masks)

    if existing_result:
        existing_result.mask_blob_path = mask_blob_path
        existing_result.coco_json_blob_path = coco_blob_path
        existing_result.batch_id = batch_id
    else:
        result = ProcessingResult(
            image_id=image.id,
            mask_blob_path=mask_blob_path,
            coco_json_blob_path=coco_blob_path,
            batch_id=batch_id,
        )
        db.add(result)

    image.processing_status = ProcessingStatus.PROCESSED
    db.commit()
    logger.info(f"Processed image {image.id} ({image.filename})")
    return True


def _process_images_background(
    image_ids: list[uuid.UUID],
    batch_id: uuid.UUID,
) -> None:
    """Background task to process images through SAM3."""
    global _processing_state

    storage = get_storage_service()
    sam3 = get_sam3_service()
    db = SessionLocal()

    try:
        sam3.load_model()

        for idx, image_id in enumerate(image_ids):
            image = db.query(Image).filter(Image.id == image_id).first()
            if not image:
                logger.warning(f"Image {image_id} not found, skipping")
                _processing_state["processed_count"] = idx + 1
                continue

            _processing_state["current_image_id"] = image_id
            _processing_state["current_image_filename"] = image.filename

            existing_result = db.query(ProcessingResult).filter(ProcessingResult.image_id == image_id).first()

            if _is_already_processed(image, existing_result):
                logger.info(f"Image {image_id} already processed, skipping")
                _processing_state["processed_count"] = idx + 1
                continue

            image.processing_status = ProcessingStatus.PROCESSING
            db.commit()

            try:
                _process_single_image(db, storage, sam3, image, batch_id, existing_result)
            except Exception as e:
                logger.error(f"Error processing image {image_id}: {e}")
                db.rollback()

            _processing_state["processed_count"] = idx + 1

    except Exception as e:
        logger.exception(f"Error in background processing: {e}")
        _processing_state["error"] = str(e)
    finally:
        sam3.unload_model()
        db.close()
        _processing_state["is_running"] = False
        _processing_state["current_image_id"] = None
        _processing_state["current_image_filename"] = None


def _is_already_processed(image: Image, existing_result: ProcessingResult | None) -> bool:
    """Check if image has already been processed."""
    return (
        image.processing_status == ProcessingStatus.PROCESSED
        and existing_result is not None
        and existing_result.mask_blob_path is not None
    )


@router.post("", response_model=ProcessResponse)
def start_processing(
    request: ProcessRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> dict:
    """Start batch processing of annotated images.

    Args:
        request: ProcessRequest containing list of image_ids.
        background_tasks: FastAPI background tasks.
        db: Database session.

    Returns:
        ProcessResponse with batch_id and message.

    Raises:
        HTTPException: If processing is already running or no valid images.
    """
    global _processing_state

    # Check if processing is already running
    if _processing_state["is_running"]:
        raise HTTPException(
            status_code=409,
            detail="Processing already in progress. Wait for it to complete.",
        )

    # Validate image_ids exist and have annotations
    valid_image_ids = []
    for image_id in request.image_ids:
        image = db.query(Image).filter(Image.id == image_id).first()
        if image:
            # Check if image has annotations
            annotation_count = db.query(Annotation).filter(Annotation.image_id == image_id).count()
            if annotation_count > 0:
                valid_image_ids.append(image_id)

    if not valid_image_ids:
        raise HTTPException(
            status_code=400,
            detail="No valid images with annotations to process.",
        )

    # Generate batch ID and initialize state
    batch_id = uuid.uuid4()
    _processing_state["batch_id"] = batch_id
    _processing_state["is_running"] = True
    _processing_state["processed_count"] = 0
    _processing_state["total_count"] = len(valid_image_ids)
    _processing_state["current_image_id"] = None
    _processing_state["current_image_filename"] = None
    _processing_state["error"] = None

    # Start background task
    background_tasks.add_task(_process_images_background, valid_image_ids, batch_id)

    return {
        "batch_id": batch_id,
        "total_images": len(valid_image_ids),
        "message": f"Processing started for {len(valid_image_ids)} images",
    }


@router.get("/status", response_model=ProcessStatus)
def get_processing_status() -> dict:
    """Get current processing status.

    Returns:
        ProcessStatus with current progress information.
    """
    return {
        "batch_id": _processing_state["batch_id"],
        "is_running": _processing_state["is_running"],
        "processed_count": _processing_state["processed_count"],
        "total_count": _processing_state["total_count"],
        "current_image_id": _processing_state["current_image_id"],
        "current_image_filename": _processing_state["current_image_filename"],
        "error": _processing_state["error"],
    }


@router.get("/mask/{image_id}")
def get_mask(
    image_id: uuid.UUID,
    db: Session = Depends(get_db),
    storage: StorageService = Depends(get_storage_service),
):
    """Get segmentation mask image for a processed image.

    Args:
        image_id: UUID of the image.
        db: Database session.
        storage: Storage service.

    Returns:
        PNG mask image.

    Raises:
        HTTPException: If image not found or not processed.
    """
    from fastapi.responses import Response

    result = db.query(ProcessingResult).filter(ProcessingResult.image_id == image_id).first()
    if not result or not result.mask_blob_path:
        raise HTTPException(status_code=404, detail="Mask not found for this image")

    try:
        mask_bytes = storage.get_image(result.mask_blob_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve mask: {e}")

    return Response(content=mask_bytes, media_type="image/png")


@router.get("/export/{image_id}")
def export_coco_json(
    image_id: uuid.UUID,
    db: Session = Depends(get_db),
    storage: StorageService = Depends(get_storage_service),
) -> dict:
    """Download COCO JSON annotation for a processed image.

    Args:
        image_id: UUID of the image to export.
        db: Database session.
        storage: Storage service.

    Returns:
        COCO JSON content as dict.

    Raises:
        HTTPException: If image not found or not processed.
    """
    from fastapi.responses import JSONResponse

    # Get image and processing result
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    result = db.query(ProcessingResult).filter(ProcessingResult.image_id == image_id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Image has not been processed yet")

    # Fetch COCO JSON from storage
    try:
        coco_bytes = storage.get_image(result.coco_json_blob_path)
        coco_json = json.loads(coco_bytes.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve COCO JSON: {e}")

    # Return as downloadable JSON with filename
    return JSONResponse(
        content=coco_json,
        headers={"Content-Disposition": f'attachment; filename="{image.filename.rsplit(".", 1)[0]}_coco.json"'},
    )
