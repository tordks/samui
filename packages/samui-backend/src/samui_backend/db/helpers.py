"""Database helper functions for common query patterns."""

import uuid

from fastapi import HTTPException
from sqlalchemy import desc
from sqlalchemy.orm import Session

from samui_backend.db.models import Image, ProcessingResult
from samui_backend.enums import SegmentationMode


def get_image_or_404(db: Session, image_id: uuid.UUID) -> Image:
    """Get an image by ID or raise 404.

    Args:
        db: Database session.
        image_id: UUID of the image.

    Returns:
        Image instance.

    Raises:
        HTTPException: 404 if image not found.
    """
    image = db.get(Image, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    return image


def get_latest_processing_result(
    db: Session,
    image_id: uuid.UUID,
    mode: SegmentationMode,
) -> ProcessingResult | None:
    """Get the latest processing result for an image and mode.

    Args:
        db: Database session.
        image_id: UUID of the image.
        mode: Segmentation mode to filter by.

    Returns:
        ProcessingResult if found, None otherwise.
    """
    return (
        db.query(ProcessingResult)
        .filter(ProcessingResult.image_id == image_id, ProcessingResult.mode == mode)
        .order_by(desc(ProcessingResult.processed_at))
        .first()
    )
