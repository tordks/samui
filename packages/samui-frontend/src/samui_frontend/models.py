"""Shared data models and enums."""

from enum import StrEnum


class SegmentationMode(StrEnum):
    """Segmentation mode for processing."""

    INSIDE_BOX = "inside_box"
    FIND_ALL = "find_all"


class PromptType(StrEnum):
    """Type of prompt/annotation."""

    SEGMENT = "segment"
    POSITIVE_EXEMPLAR = "positive_exemplar"
    NEGATIVE_EXEMPLAR = "negative_exemplar"


class AnnotationSource(StrEnum):
    """Source of an annotation."""

    USER = "user"
    MODEL = "model"
