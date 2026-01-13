"""Processing routes for SAM3 inference and COCO export.

Note: Job creation and status is handled by routes/jobs.py.
This module provides endpoints for mask/export retrieval by image_id.
"""

import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, Response
from sqlalchemy import desc
from sqlalchemy.orm import Session

from samui_backend.db.database import get_db
from samui_backend.db.models import Image, ProcessingResult
from samui_backend.dependencies import get_storage_service
from samui_backend.enums import SegmentationMode
from samui_backend.services.storage import StorageService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/process", tags=["processing"])


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
