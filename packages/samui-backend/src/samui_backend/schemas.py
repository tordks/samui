"""Pydantic schemas for API request/response models."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from samui_backend.enums import (
    PromptType,
    SegmentationMode,
)


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
    text_prompt: str | None = None


class ImageUpdate(BaseModel):
    """Schema for updating an image."""

    text_prompt: str | None = None


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
    prompt_type: PromptType = PromptType.SEGMENT


class AnnotationResponse(BaseModel):
    """Schema for annotation response."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    image_id: uuid.UUID
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    prompt_type: PromptType
    created_at: datetime


class AnnotationList(BaseModel):
    """Schema for list of annotations response."""

    annotations: list[AnnotationResponse]
    total: int


class ProcessRequest(BaseModel):
    """Schema for requesting processing of images."""

    image_ids: list[uuid.UUID]
    mode: SegmentationMode = SegmentationMode.INSIDE_BOX


class ProcessResponse(BaseModel):
    """Schema for processing start response."""

    batch_id: uuid.UUID
    total_images: int
    message: str


class ProcessStatus(BaseModel):
    """Schema for processing status response."""

    batch_id: uuid.UUID | None
    is_running: bool
    processed_count: int
    total_count: int
    current_image_id: uuid.UUID | None
    current_image_filename: str | None
    error: str | None


class ProcessingResultResponse(BaseModel):
    """Schema for processing result response."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    job_id: uuid.UUID
    image_id: uuid.UUID
    mode: SegmentationMode
    mask_blob_path: str
    coco_json_blob_path: str
    processed_at: datetime
    annotation_ids: list[str] | None = None
    text_prompt_used: str | None = None
    bboxes: list[dict] | None = None
