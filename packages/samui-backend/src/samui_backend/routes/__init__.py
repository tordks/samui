"""API routes."""

from samui_backend.routes.annotations import router as annotations_router
from samui_backend.routes.images import router as images_router
from samui_backend.routes.processing import router as processing_router

__all__ = ["annotations_router", "images_router", "processing_router"]
