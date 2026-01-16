"""Annotation CRUD endpoints."""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from samui_backend.db.database import get_db
from samui_backend.db.helpers import get_image_or_404
from samui_backend.db.models import BboxAnnotation, PointAnnotation
from samui_backend.enums import PromptType
from samui_backend.schemas import (
    BboxAnnotationCreate,
    BboxAnnotationList,
    BboxAnnotationResponse,
    PointAnnotationCreate,
    PointAnnotationList,
    PointAnnotationResponse,
)

router = APIRouter(prefix="/annotations", tags=["annotations"])


@router.post("", response_model=BboxAnnotationResponse, status_code=201)
def create_annotation(
    annotation: BboxAnnotationCreate,
    db: Session = Depends(get_db),
) -> BboxAnnotation:
    """Create a bounding box annotation for an image."""
    image = get_image_or_404(db, annotation.image_id)

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
    db_annotation = BboxAnnotation(
        image_id=annotation.image_id,
        bbox_x=annotation.bbox_x,
        bbox_y=annotation.bbox_y,
        bbox_width=annotation.bbox_width,
        bbox_height=annotation.bbox_height,
        prompt_type=annotation.prompt_type,
    )
    db.add(db_annotation)
    db.commit()
    db.refresh(db_annotation)

    return db_annotation


@router.get("/{image_id}", response_model=BboxAnnotationList)
def get_annotations(
    image_id: uuid.UUID,
    prompt_type: PromptType | None = None,
    db: Session = Depends(get_db),
) -> dict:
    """Get all annotations for an image, optionally filtered by prompt_type."""
    get_image_or_404(db, image_id)

    query = db.query(BboxAnnotation).filter(BboxAnnotation.image_id == image_id)
    if prompt_type is not None:
        query = query.filter(BboxAnnotation.prompt_type == prompt_type)

    annotations = query.order_by(BboxAnnotation.created_at.asc()).all()
    return {"annotations": annotations, "total": len(annotations)}


@router.delete("/{annotation_id}", status_code=204)
def delete_annotation(
    annotation_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> None:
    """Delete an annotation."""
    annotation = db.query(BboxAnnotation).filter(BboxAnnotation.id == annotation_id).first()
    if not annotation:
        raise HTTPException(status_code=404, detail="Annotation not found")

    db.delete(annotation)
    db.commit()


# Point annotation router
point_router = APIRouter(prefix="/point-annotations", tags=["point-annotations"])


@point_router.post("", response_model=PointAnnotationResponse, status_code=201)
def create_point_annotation(
    annotation: PointAnnotationCreate,
    db: Session = Depends(get_db),
) -> PointAnnotation:
    """Create a point annotation for an image."""
    image = get_image_or_404(db, annotation.image_id)

    # Validate point coordinates are within image bounds
    if annotation.point_x < 0 or annotation.point_y < 0:
        raise HTTPException(status_code=400, detail="Point coordinates must be non-negative")

    if annotation.point_x >= image.width:
        raise HTTPException(status_code=400, detail="Point x coordinate exceeds image width")

    if annotation.point_y >= image.height:
        raise HTTPException(status_code=400, detail="Point y coordinate exceeds image height")

    # Create annotation
    db_annotation = PointAnnotation(
        image_id=annotation.image_id,
        point_x=annotation.point_x,
        point_y=annotation.point_y,
        is_positive=annotation.is_positive,
    )
    db.add(db_annotation)
    db.commit()
    db.refresh(db_annotation)

    return db_annotation


@point_router.get("/{image_id}", response_model=PointAnnotationList)
def get_point_annotations(
    image_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> dict:
    """Get all point annotations for an image."""
    get_image_or_404(db, image_id)

    annotations = (
        db.query(PointAnnotation)
        .filter(PointAnnotation.image_id == image_id)
        .order_by(PointAnnotation.created_at.asc())
        .all()
    )
    return {"annotations": annotations, "total": len(annotations)}


@point_router.delete("/{annotation_id}", status_code=204)
def delete_point_annotation(
    annotation_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> None:
    """Delete a point annotation."""
    annotation = db.query(PointAnnotation).filter(PointAnnotation.id == annotation_id).first()
    if not annotation:
        raise HTTPException(status_code=404, detail="Point annotation not found")

    db.delete(annotation)
    db.commit()
