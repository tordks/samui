"""Pydantic schemas for API request/response models."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from samui_backend.db.models import ProcessingStatus


class ImageCreate(BaseModel):
    """Schema for creating an image (internal use)."""

    filename: str
    blob_path: str
    width: int
    height: int


class ImageResponse(BaseModel):
    """Schema for image response."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    blob_path: str
    width: int
    height: int
    created_at: datetime
    processing_status: ProcessingStatus


class ImageList(BaseModel):
    """Schema for list of images response."""

    images: list[ImageResponse]
    total: int


class AnnotationCreate(BaseModel):
    """Schema for creating an annotation."""

    image_id: uuid.UUID
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int


class AnnotationResponse(BaseModel):
    """Schema for annotation response."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    image_id: uuid.UUID
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    created_at: datetime


class AnnotationList(BaseModel):
    """Schema for list of annotations response."""

    annotations: list[AnnotationResponse]
    total: int
