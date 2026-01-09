"""Shared enums for the application."""

from enum import StrEnum


class ProcessingStatus(StrEnum):
    """Processing status for images."""

    PENDING = "pending"
    ANNOTATED = "annotated"
    PROCESSING = "processing"
    PROCESSED = "processed"


class PromptType(StrEnum):
    """Type of annotation prompt for segmentation."""

    SEGMENT = "segment"
    POSITIVE_EXEMPLAR = "positive_exemplar"
    NEGATIVE_EXEMPLAR = "negative_exemplar"


class AnnotationSource(StrEnum):
    """Source of an annotation."""

    USER = "user"
    MODEL = "model"


class SegmentationMode(StrEnum):
    """Segmentation mode for processing."""

    INSIDE_BOX = "inside_box"
    FIND_ALL = "find_all"
