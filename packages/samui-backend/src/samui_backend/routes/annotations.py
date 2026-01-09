"""Annotation CRUD endpoints."""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from samui_backend.db.database import get_db
from samui_backend.db.models import Annotation, Image, ProcessingStatus, PromptType
from samui_backend.schemas import AnnotationCreate, AnnotationList, AnnotationResponse

router = APIRouter(prefix="/annotations", tags=["annotations"])


@router.post("", response_model=AnnotationResponse, status_code=201)
def create_annotation(
    annotation: AnnotationCreate,
    db: Session = Depends(get_db),
) -> Annotation:
    """Create a bounding box annotation for an image."""
    # Verify image exists
    image = db.query(Image).filter(Image.id == annotation.image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Validate bbox dimensions
    if annotation.bbox_width <= 0 or annotation.bbox_height <= 0:
        raise HTTPException(status_code=400, detail="Bounding box must have positive dimensions")

    if annotation.bbox_x < 0 or annotation.bbox_y < 0:
        raise HTTPException(status_code=400, detail="Bounding box coordinates must be non-negative")

    if annotation.bbox_x + annotation.bbox_width > image.width:
        raise HTTPException(status_code=400, detail="Bounding box exceeds image width")

    if annotation.bbox_y + annotation.bbox_height > image.height:
        raise HTTPException(status_code=400, detail="Bounding box exceeds image height")

    # Create annotation
    db_annotation = Annotation(
        image_id=annotation.image_id,
        bbox_x=annotation.bbox_x,
        bbox_y=annotation.bbox_y,
        bbox_width=annotation.bbox_width,
        bbox_height=annotation.bbox_height,
        prompt_type=annotation.prompt_type,
    )
    db.add(db_annotation)

    # Update image status to annotated if this is the first annotation
    if image.processing_status == ProcessingStatus.PENDING:
        image.processing_status = ProcessingStatus.ANNOTATED

    db.commit()
    db.refresh(db_annotation)

    return db_annotation


@router.get("/{image_id}", response_model=AnnotationList)
def get_annotations(
    image_id: uuid.UUID,
    prompt_type: PromptType | None = None,
    db: Session = Depends(get_db),
) -> dict:
    """Get all annotations for an image, optionally filtered by prompt_type."""
    # Verify image exists
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    query = db.query(Annotation).filter(Annotation.image_id == image_id)
    if prompt_type is not None:
        query = query.filter(Annotation.prompt_type == prompt_type)

    annotations = query.order_by(Annotation.created_at.asc()).all()
    return {"annotations": annotations, "total": len(annotations)}


@router.delete("/{annotation_id}", status_code=204)
def delete_annotation(
    annotation_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> None:
    """Delete an annotation."""
    annotation = db.query(Annotation).filter(Annotation.id == annotation_id).first()
    if not annotation:
        raise HTTPException(status_code=404, detail="Annotation not found")

    db.delete(annotation)
    db.commit()
