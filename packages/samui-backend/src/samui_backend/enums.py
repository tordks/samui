"""Shared enums for the application."""

from enum import StrEnum


class JobStatus(StrEnum):
    """Status of a processing job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PromptType(StrEnum):
    """Type of annotation prompt for segmentation."""

    SEGMENT = "segment"
    POSITIVE_EXEMPLAR = "positive_exemplar"
    NEGATIVE_EXEMPLAR = "negative_exemplar"


class SegmentationMode(StrEnum):
    """Segmentation mode for processing."""

    INSIDE_BOX = "inside_box"
    FIND_ALL = "find_all"
