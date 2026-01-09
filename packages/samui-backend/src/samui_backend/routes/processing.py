"""Processing routes for SAM3 inference and COCO export."""

import json
import logging
import uuid
from io import BytesIO

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import JSONResponse, Response
from numpy.typing import NDArray
from PIL import Image as PILImage
from sqlalchemy.orm import Session

from samui_backend.db.database import SessionLocal, get_db
from samui_backend.db.models import (
    Annotation,
    AnnotationSource,
    Image,
    ProcessingResult,
    ProcessingStatus,
    PromptType,
    SegmentationMode,
)
from samui_backend.dependencies import get_sam3_service, get_storage_service
from samui_backend.schemas import ProcessRequest, ProcessResponse, ProcessStatus
from samui_backend.services import SAM3Service, StorageService, generate_coco_json

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/process", tags=["processing"])

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


def _save_mask_to_storage(
    storage: StorageService,
    masks: NDArray[np.uint8],
    image_id: uuid.UUID,
    mode: SegmentationMode = SegmentationMode.INSIDE_BOX,
) -> str:
    """Save combined mask image to storage."""
    mode_suffix = f"_{mode.value}" if mode != SegmentationMode.INSIDE_BOX else ""
    mask_blob_path = f"masks/{image_id}{mode_suffix}.png"
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
    mode: SegmentationMode = SegmentationMode.INSIDE_BOX,
) -> str:
    """Generate and save COCO JSON to storage."""
    mode_suffix = f"_{mode.value}" if mode != SegmentationMode.INSIDE_BOX else ""
    coco_blob_path = f"coco/{image.id}{mode_suffix}.json"
    coco_json = generate_coco_json(
        image_id=image.id,
        filename=image.filename,
        width=image.width,
        height=image.height,
        bboxes=bboxes,
        masks=masks,
    )
    coco_bytes = json.dumps(coco_json, indent=2).encode("utf-8")
    storage.upload_blob(coco_blob_path, coco_bytes, content_type="application/json")
    return coco_blob_path


def _save_processing_result(
    db: Session,
    image_id: uuid.UUID,
    mode: SegmentationMode,
    mask_blob_path: str,
    coco_blob_path: str,
    batch_id: uuid.UUID,
    existing_result: ProcessingResult | None,
) -> None:
    """Update existing or create new ProcessingResult."""
    if existing_result:
        existing_result.mask_blob_path = mask_blob_path
        existing_result.coco_json_blob_path = coco_blob_path
        existing_result.batch_id = batch_id
    else:
        db.add(
            ProcessingResult(
                image_id=image_id,
                mode=mode,
                mask_blob_path=mask_blob_path,
                coco_json_blob_path=coco_blob_path,
                batch_id=batch_id,
            )
        )


def _process_single_image(
    db: Session,
    storage: StorageService,
    sam3: SAM3Service,
    image: Image,
    batch_id: uuid.UUID,
    existing_result: ProcessingResult | None,
    mode: SegmentationMode = SegmentationMode.INSIDE_BOX,
) -> bool:
    """Process a single image through SAM3 inference (inside-box mode).

    Returns:
        True if processed successfully, False otherwise.
    """
    image_data = storage.get_image(image.blob_path)
    pil_image = PILImage.open(BytesIO(image_data)).convert("RGB")

    # For inside-box mode, only use SEGMENT annotations
    annotations = (
        db.query(Annotation).filter(Annotation.image_id == image.id, Annotation.prompt_type == PromptType.SEGMENT).all()
    )
    if not annotations:
        logger.warning(f"No segment annotations for image {image.id}, skipping")
        image.processing_status = ProcessingStatus.ANNOTATED
        db.commit()
        return False

    bboxes = [(ann.bbox_x, ann.bbox_y, ann.bbox_width, ann.bbox_height) for ann in annotations]

    masks = sam3.process_image(pil_image, bboxes)
    mask_blob_path = _save_mask_to_storage(storage, masks, image.id, mode)
    coco_blob_path = _save_coco_to_storage(storage, image, bboxes, masks, mode)

    _save_processing_result(db, image.id, mode, mask_blob_path, coco_blob_path, batch_id, existing_result)

    image.processing_status = ProcessingStatus.PROCESSED
    db.commit()
    logger.info(f"Processed image {image.id} ({image.filename}) with mode {mode.value}")
    return True


def _process_single_image_find_all(
    db: Session,
    storage: StorageService,
    sam3: SAM3Service,
    image: Image,
    batch_id: uuid.UUID,
    existing_result: ProcessingResult | None,
) -> bool:
    """Process a single image through SAM3 find-all inference.

    Uses text prompts and/or exemplar boxes to discover all matching objects.
    Creates new annotations for discovered objects.

    Returns:
        True if processed successfully, False otherwise.
    """
    # Load text prompt from image
    text_prompt = image.text_prompt

    # Load exemplar annotations (positive and negative)
    exemplar_annotations = (
        db.query(Annotation)
        .filter(
            Annotation.image_id == image.id,
            Annotation.prompt_type.in_([PromptType.POSITIVE_EXEMPLAR, PromptType.NEGATIVE_EXEMPLAR]),
        )
        .all()
    )

    # Validate: need text prompt or exemplars
    if not text_prompt and not exemplar_annotations:
        logger.warning(f"No text prompt or exemplars for image {image.id} in find-all mode, skipping")
        image.processing_status = ProcessingStatus.ANNOTATED
        db.commit()
        return False

    # Load image data
    image_data = storage.get_image(image.blob_path)
    pil_image = PILImage.open(BytesIO(image_data)).convert("RGB")

    # Convert exemplar annotations to (bbox_xywh, is_positive) format
    exemplar_boxes = None
    if exemplar_annotations:
        exemplar_boxes = [
            (
                (ann.bbox_x, ann.bbox_y, ann.bbox_width, ann.bbox_height),
                ann.prompt_type == PromptType.POSITIVE_EXEMPLAR,
            )
            for ann in exemplar_annotations
        ]

    # Run find-all inference
    find_result = sam3.process_image_find_all(pil_image, text_prompt, exemplar_boxes)

    # Create annotations for discovered objects
    for x, y, w, h in find_result.bboxes:
        db.add(
            Annotation(
                image_id=image.id,
                bbox_x=x,
                bbox_y=y,
                bbox_width=w,
                bbox_height=h,
                prompt_type=PromptType.SEGMENT,
                source=AnnotationSource.MODEL,
            )
        )

    # Save masks and COCO JSON
    mode = SegmentationMode.FIND_ALL
    if find_result.masks.size > 0:
        mask_blob_path = _save_mask_to_storage(storage, find_result.masks, image.id, mode)
        coco_blob_path = _save_coco_to_storage(storage, image, find_result.bboxes, find_result.masks, mode)
    else:
        # No discoveries - save empty results
        empty_masks = np.zeros((0, pil_image.height, pil_image.width), dtype=np.uint8)
        mask_blob_path = _save_mask_to_storage(storage, empty_masks, image.id, mode)
        coco_blob_path = _save_coco_to_storage(storage, image, [], empty_masks, mode)

    _save_processing_result(db, image.id, mode, mask_blob_path, coco_blob_path, batch_id, existing_result)

    image.processing_status = ProcessingStatus.PROCESSED
    db.commit()
    logger.info(
        f"Processed image {image.id} ({image.filename}) with find-all mode, "
        f"discovered {len(find_result.bboxes)} objects"
    )
    return True


def _process_images_background(
    image_ids: list[uuid.UUID],
    batch_id: uuid.UUID,
    mode: SegmentationMode = SegmentationMode.INSIDE_BOX,
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

            # Check for existing result with the same mode
            existing_result = (
                db.query(ProcessingResult)
                .filter(ProcessingResult.image_id == image_id, ProcessingResult.mode == mode)
                .first()
            )

            if _is_already_processed(existing_result, mode):
                logger.info(f"Image {image_id} already processed with mode {mode.value}, skipping")
                _processing_state["processed_count"] = idx + 1
                continue

            image.processing_status = ProcessingStatus.PROCESSING
            db.commit()

            try:
                # Route to appropriate processing function based on mode
                if mode == SegmentationMode.INSIDE_BOX:
                    _process_single_image(db, storage, sam3, image, batch_id, existing_result, mode)
                else:
                    _process_single_image_find_all(db, storage, sam3, image, batch_id, existing_result)
            except Exception as e:
                logger.error(f"Error processing image {image_id} with mode {mode.value}: {e}")
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


def _is_already_processed(
    existing_result: ProcessingResult | None,
    mode: SegmentationMode = SegmentationMode.INSIDE_BOX,
) -> bool:
    """Check if image has already been processed for the given mode."""
    return existing_result is not None and existing_result.mode == mode and existing_result.mask_blob_path is not None


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

    # Validate image_ids exist and have appropriate prompts for the mode
    mode = request.mode
    valid_image_ids = []

    for image_id in request.image_ids:
        image = db.query(Image).filter(Image.id == image_id).first()
        if not image:
            continue

        if mode == SegmentationMode.INSIDE_BOX:
            # Inside-box mode requires SEGMENT annotations
            segment_count = (
                db.query(Annotation)
                .filter(Annotation.image_id == image_id, Annotation.prompt_type == PromptType.SEGMENT)
                .count()
            )
            if segment_count > 0:
                valid_image_ids.append(image_id)
        else:
            # Find-all mode requires text_prompt OR exemplar annotations
            has_text = image.text_prompt is not None and image.text_prompt.strip() != ""
            exemplar_count = (
                db.query(Annotation)
                .filter(
                    Annotation.image_id == image_id,
                    Annotation.prompt_type.in_([PromptType.POSITIVE_EXEMPLAR, PromptType.NEGATIVE_EXEMPLAR]),
                )
                .count()
            )
            if has_text or exemplar_count > 0:
                valid_image_ids.append(image_id)

    if not valid_image_ids:
        if mode == SegmentationMode.INSIDE_BOX:
            detail = "No valid images with segment annotations to process."
        else:
            detail = "No valid images with text prompts or exemplar boxes to process."
        raise HTTPException(status_code=400, detail=detail)

    # Generate batch ID and initialize state
    batch_id = uuid.uuid4()
    _processing_state["batch_id"] = batch_id
    _processing_state["is_running"] = True
    _processing_state["processed_count"] = 0
    _processing_state["total_count"] = len(valid_image_ids)
    _processing_state["current_image_id"] = None
    _processing_state["current_image_filename"] = None
    _processing_state["error"] = None

    # Start background task with mode
    background_tasks.add_task(_process_images_background, valid_image_ids, batch_id, mode)

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
    mode: SegmentationMode = SegmentationMode.INSIDE_BOX,
    db: Session = Depends(get_db),
    storage: StorageService = Depends(get_storage_service),
):
    """Get segmentation mask image for a processed image.

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
    result = (
        db.query(ProcessingResult).filter(ProcessingResult.image_id == image_id, ProcessingResult.mode == mode).first()
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
) -> dict:
    """Download COCO JSON annotation for a processed image.

    Args:
        image_id: UUID of the image to export.
        mode: Segmentation mode (defaults to INSIDE_BOX for backward compatibility).
        db: Database session.
        storage: Storage service.

    Returns:
        COCO JSON content as dict.

    Raises:
        HTTPException: If image not found or not processed.
    """
    # Get image and processing result
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    result = (
        db.query(ProcessingResult).filter(ProcessingResult.image_id == image_id, ProcessingResult.mode == mode).first()
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
