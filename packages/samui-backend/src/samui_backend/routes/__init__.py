"""API routes."""

from samui_backend.routes.annotations import router as annotations_router
from samui_backend.routes.images import router as images_router

__all__ = ["annotations_router", "images_router"]
