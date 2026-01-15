"""Pydantic schemas for API request/response models."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, computed_field

from samui_backend.enums import (
    JobStatus,
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


class ProcessingJobCreate(BaseModel):
    """Schema for creating a processing job."""

    image_ids: list[uuid.UUID]
    mode: SegmentationMode = SegmentationMode.INSIDE_BOX
    force_all: bool = False


class ProcessingJobResponse(BaseModel):
    """Schema for processing job response."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    mode: SegmentationMode
    status: JobStatus
    image_ids: list[str]
    image_filenames: list[str]
    current_index: int
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def image_count(self) -> int:
        """Return the total number of images in the job."""
        return len(self.image_ids)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_running(self) -> bool:
        """Return whether the job is currently running."""
        return self.status == JobStatus.RUNNING

    @computed_field  # type: ignore[prop-decorator]
    @property
    def processed_count(self) -> int:
        """Return the number of images processed so far."""
        return self.current_index

    @computed_field  # type: ignore[prop-decorator]
    @property
    def current_image_filename(self) -> str | None:
        """Return the filename of the image currently being processed."""
        if self.is_running and self.current_index < len(self.image_filenames):
            return self.image_filenames[self.current_index]
        return None


class ProcessingHistoryResponse(BaseModel):
    """Schema for processing history response (for image history page)."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    job_id: uuid.UUID
    image_id: uuid.UUID
    mode: SegmentationMode
    processed_at: datetime
    text_prompt_used: str | None = None
    bboxes: list[dict] | None = None
    mask_blob_path: str
