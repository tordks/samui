"""SQLAlchemy database models."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from samui_backend.db.database import Base
from samui_backend.enums import JobStatus, PromptType, SegmentationMode

# Re-export enums for backward compatibility
__all__ = [
    "JobStatus",
    "PromptType",
    "SegmentationMode",
    "Image",
    "BboxAnnotation",
    "PointAnnotation",
    "ProcessingResult",
    "ProcessingJob",
]


class Image(Base):
    """Image metadata model."""

    __tablename__ = "images"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    blob_path: Mapped[str] = mapped_column(String(512), nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    text_prompt: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    bbox_annotations: Mapped[list["BboxAnnotation"]] = relationship(
        "BboxAnnotation", back_populates="image", cascade="all, delete-orphan"
    )
    point_annotations: Mapped[list["PointAnnotation"]] = relationship(
        "PointAnnotation", back_populates="image", cascade="all, delete-orphan"
    )
    processing_results: Mapped[list["ProcessingResult"]] = relationship(
        "ProcessingResult", back_populates="image", cascade="all, delete-orphan"
    )


class BboxAnnotation(Base):
    """Bounding box annotation model."""

    __tablename__ = "bbox_annotations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("images.id", ondelete="CASCADE"), nullable=False
    )
    bbox_x: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_y: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_width: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_height: Mapped[int] = mapped_column(Integer, nullable=False)
    prompt_type: Mapped[PromptType] = mapped_column(Enum(PromptType), default=PromptType.SEGMENT, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    image: Mapped["Image"] = relationship("Image", back_populates="bbox_annotations")


class PointAnnotation(Base):
    """Point annotation model for point-based segmentation."""

    __tablename__ = "point_annotations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("images.id", ondelete="CASCADE"), nullable=False
    )
    point_x: Mapped[int] = mapped_column(Integer, nullable=False)
    point_y: Mapped[int] = mapped_column(Integer, nullable=False)
    is_positive: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    image: Mapped["Image"] = relationship("Image", back_populates="point_annotations")


class ProcessingJob(Base):
    """Processing job model tracking batch processing with queue support."""

    __tablename__ = "processing_jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    mode: Mapped[SegmentationMode] = mapped_column(
        Enum(SegmentationMode), default=SegmentationMode.INSIDE_BOX, nullable=False
    )
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.QUEUED, nullable=False)
    image_ids: Mapped[list] = mapped_column(JSON, nullable=False)  # list of UUID strings
    image_filenames: Mapped[list] = mapped_column(JSON, nullable=False)  # list of filenames
    current_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    annotations_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # dict[str, AnnotationsSnapshot]

    results: Mapped[list["ProcessingResult"]] = relationship("ProcessingResult", back_populates="job")


class ProcessingResult(Base):
    """Processing result model storing mask and COCO JSON paths."""

    __tablename__ = "processing_results"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("processing_jobs.id", ondelete="CASCADE"),
        nullable=False,
    )
    image_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("images.id", ondelete="CASCADE"),
        nullable=False,
    )
    mode: Mapped[SegmentationMode] = mapped_column(
        Enum(SegmentationMode), default=SegmentationMode.INSIDE_BOX, nullable=False
    )
    mask_blob_path: Mapped[str] = mapped_column(String(512), nullable=False)
    coco_json_blob_path: Mapped[str] = mapped_column(String(512), nullable=False)
    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    bboxes: Mapped[list | None] = mapped_column(JSON, nullable=True)  # list of {x, y, width, height}

    job: Mapped["ProcessingJob"] = relationship("ProcessingJob", back_populates="results")
    image: Mapped["Image"] = relationship("Image", back_populates="processing_results")
