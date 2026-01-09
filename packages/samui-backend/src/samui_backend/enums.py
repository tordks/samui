"""Shared enums for the application."""

import enum


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
