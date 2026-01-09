"""SQLAlchemy database models."""

import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from samui_backend.db.database import Base


class ProcessingStatus(str, enum.Enum):
    """Processing status for images."""

    PENDING = "pending"
    ANNOTATED = "annotated"
    PROCESSING = "processing"
    PROCESSED = "processed"


class PromptType(str, enum.Enum):
    """Type of annotation prompt for segmentation."""

    SEGMENT = "segment"
    POSITIVE_EXEMPLAR = "positive_exemplar"
    NEGATIVE_EXEMPLAR = "negative_exemplar"


class AnnotationSource(str, enum.Enum):
    """Source of an annotation."""

    USER = "user"
    MODEL = "model"


class SegmentationMode(str, enum.Enum):
    """Segmentation mode for processing."""

    INSIDE_BOX = "inside_box"
    FIND_ALL = "find_all"


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
    processing_status: Mapped[ProcessingStatus] = mapped_column(
        Enum(ProcessingStatus), default=ProcessingStatus.PENDING, nullable=False
    )
    text_prompt: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    annotations: Mapped[list["Annotation"]] = relationship(
        "Annotation", back_populates="image", cascade="all, delete-orphan"
    )
    processing_result: Mapped["ProcessingResult | None"] = relationship(
        "ProcessingResult", back_populates="image", cascade="all, delete-orphan", uselist=False
    )


class Annotation(Base):
    """Bounding box annotation model."""

    __tablename__ = "annotations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("images.id", ondelete="CASCADE"), nullable=False
    )
    bbox_x: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_y: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_width: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_height: Mapped[int] = mapped_column(Integer, nullable=False)
    prompt_type: Mapped[PromptType] = mapped_column(Enum(PromptType), default=PromptType.SEGMENT, nullable=False)
    source: Mapped[AnnotationSource] = mapped_column(
        Enum(AnnotationSource), default=AnnotationSource.USER, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    image: Mapped["Image"] = relationship("Image", back_populates="annotations")


class ProcessingResult(Base):
    """Processing result model storing mask and COCO JSON paths."""

    __tablename__ = "processing_results"
    __table_args__ = (UniqueConstraint("image_id", "mode", name="uq_processing_result_image_mode"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
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
    batch_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)

    image: Mapped["Image"] = relationship("Image", back_populates="processing_result")
