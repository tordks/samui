"""Image upload, list, and delete endpoints."""

import logging
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response
from sqlalchemy import desc
from sqlalchemy.orm import Session

from samui_backend.db.database import get_db
from samui_backend.db.helpers import get_image_or_404
from samui_backend.db.models import Image, ProcessingResult
from samui_backend.dependencies import get_storage_service
from samui_backend.enums import SegmentationMode
from samui_backend.schemas import ImageList, ImageResponse, ImageUpdate, ProcessingHistoryResponse
from samui_backend.services.storage import StorageService
from samui_backend.utils import get_image_content_type

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/images", tags=["images"])


@router.post("", response_model=ImageResponse, status_code=201)
async def upload_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    storage: StorageService = Depends(get_storage_service),
) -> Image:
    """Upload an image to storage and save metadata to database."""
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read file content
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    # Upload to storage
    try:
        blob_path, width, height = storage.upload_image(content, file.filename or "image.jpg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload image: {e}") from None

    # Save metadata to database
    image = Image(
        filename=file.filename or "image.jpg",
        blob_path=blob_path,
        width=width,
        height=height,
    )
    db.add(image)
    db.commit()
    db.refresh(image)

    return image


@router.get("", response_model=ImageList)
def list_images(db: Session = Depends(get_db)) -> dict:
    """List all images."""
    images = db.query(Image).order_by(Image.created_at.desc()).all()
    return {"images": images, "total": len(images)}


@router.get("/history", response_model=list[ProcessingHistoryResponse])
def get_all_history(
    mode: SegmentationMode = SegmentationMode.INSIDE_BOX,
    db: Session = Depends(get_db),
) -> list[ProcessingResult]:
    """Get all processing history across all images.

    Args:
        mode: Segmentation mode to filter by.
        db: Database session.

    Returns:
        List of ProcessingHistoryResponse, newest first.
    """
    results = (
        db.query(ProcessingResult)
        .filter(ProcessingResult.mode == mode)
        .order_by(desc(ProcessingResult.processed_at))
        .all()
    )
    return results


@router.get("/{image_id}", response_model=ImageResponse)
def get_image(image_id: uuid.UUID, db: Session = Depends(get_db)) -> Image:
    """Get a single image by ID."""
    return get_image_or_404(db, image_id)


@router.patch("/{image_id}", response_model=ImageResponse)
def update_image(
    image_id: uuid.UUID,
    update: ImageUpdate,
    db: Session = Depends(get_db),
) -> Image:
    """Update image metadata."""
    image = get_image_or_404(db, image_id)

    if "text_prompt" in update.model_fields_set:
        image.text_prompt = update.text_prompt

    db.commit()
    db.refresh(image)
    return image


@router.get("/{image_id}/data")
def get_image_data(
    image_id: uuid.UUID,
    db: Session = Depends(get_db),
    storage: StorageService = Depends(get_storage_service),
) -> bytes:
    """Get the actual image data."""
    image = get_image_or_404(db, image_id)

    try:
        data = storage.get_image(image.blob_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve image: {e}") from None

    # Determine content type from filename
    extension = image.filename.rsplit(".", 1)[-1] if "." in image.filename else "jpg"
    content_type = get_image_content_type(extension)

    return Response(content=data, media_type=content_type)


@router.delete("/{image_id}", status_code=204)
def delete_image(
    image_id: uuid.UUID,
    db: Session = Depends(get_db),
    storage: StorageService = Depends(get_storage_service),
) -> None:
    """Delete an image and its metadata."""
    image = get_image_or_404(db, image_id)

    # Delete from storage
    try:
        storage.delete_image(image.blob_path)
    except Exception as e:
        logger.warning(f"Failed to delete blob {image.blob_path}: {e}")

    # Delete from database
    db.delete(image)
    db.commit()


@router.get("/{image_id}/history", response_model=list[ProcessingHistoryResponse])
def get_image_history(
    image_id: uuid.UUID,
    mode: SegmentationMode = SegmentationMode.INSIDE_BOX,
    db: Session = Depends(get_db),
) -> list[ProcessingResult]:
    """Get processing history for an image.

    Args:
        image_id: UUID of the image.
        mode: Segmentation mode to filter by.
        db: Database session.

    Returns:
        List of ProcessingHistoryResponse, newest first.

    Raises:
        HTTPException: If image not found.
    """
    get_image_or_404(db, image_id)

    results = (
        db.query(ProcessingResult)
        .filter(ProcessingResult.image_id == image_id, ProcessingResult.mode == mode)
        .order_by(desc(ProcessingResult.processed_at))
        .all()
    )
    return results
