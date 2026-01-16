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


class BboxAnnotationCreate(BaseModel):
    """Schema for creating a bounding box annotation."""

    image_id: uuid.UUID
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    prompt_type: PromptType = PromptType.SEGMENT


class BboxAnnotationResponse(BaseModel):
    """Schema for bounding box annotation response."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    image_id: uuid.UUID
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    prompt_type: PromptType
    created_at: datetime


class BboxAnnotationList(BaseModel):
    """Schema for list of bounding box annotations response."""

    annotations: list[BboxAnnotationResponse]
    total: int


class PointAnnotationCreate(BaseModel):
    """Schema for creating a point annotation."""

    image_id: uuid.UUID
    point_x: int
    point_y: int
    is_positive: bool = True


class PointAnnotationResponse(BaseModel):
    """Schema for point annotation response."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    image_id: uuid.UUID
    point_x: int
    point_y: int
    is_positive: bool
    created_at: datetime


class PointAnnotationList(BaseModel):
    """Schema for list of point annotations response."""

    annotations: list[PointAnnotationResponse]
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
    bboxes: list[dict] | None = None


class BboxAnnotationSnapshot(BaseModel):
    """Snapshot of a bbox annotation for job processing."""

    id: uuid.UUID
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    prompt_type: PromptType


class PointAnnotationSnapshot(BaseModel):
    """Snapshot of a point annotation for job processing."""

    id: uuid.UUID
    point_x: int
    point_y: int
    is_positive: bool


class AnnotationsSnapshot(BaseModel):
    """Snapshot of an image's annotations at job submission time."""

    text_prompt: str | None = None
    bbox_annotations: list[BboxAnnotationSnapshot] = []
    point_annotations: list[PointAnnotationSnapshot] = []


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
    bboxes: list[dict] | None = None
    mask_blob_path: str
