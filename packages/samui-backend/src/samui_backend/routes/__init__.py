"""API routes."""

from samui_backend.routes.annotations import point_router as point_annotations_router
from samui_backend.routes.annotations import router as annotations_router
from samui_backend.routes.images import router as images_router
from samui_backend.routes.jobs import results_router
from samui_backend.routes.jobs import router as jobs_router
from samui_backend.routes.processing import router as processing_router

__all__ = [
    "annotations_router",
    "images_router",
    "jobs_router",
    "point_annotations_router",
    "processing_router",
    "results_router",
]
