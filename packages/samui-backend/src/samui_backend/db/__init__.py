"""Database module."""

from samui_backend.db.database import Base, SessionLocal, engine, get_db

# Import helpers after models to avoid circular imports
from samui_backend.db.helpers import get_image_or_404, get_latest_processing_result
from samui_backend.db.models import Image, JobStatus, ProcessingJob, ProcessingResult

__all__ = [
    "Base",
    "SessionLocal",
    "engine",
    "get_db",
    "get_image_or_404",
    "get_latest_processing_result",
    "Image",
    "JobStatus",
    "ProcessingJob",
    "ProcessingResult",
]
