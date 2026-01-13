"""FastAPI application entry point."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from samui_backend.db.database import Base, SessionLocal, engine
from samui_backend.routes import annotations_router, images_router, jobs_router, processing_router, results_router
from samui_backend.services import cleanup_stale_jobs

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: create database tables on startup, cleanup stale jobs."""
    Base.metadata.create_all(bind=engine)

    # Cleanup any jobs that were running when the server stopped
    db = SessionLocal()
    try:
        count = cleanup_stale_jobs(db)
        if count > 0:
            logger.info(f"Cleaned up {count} stale jobs on startup")
    finally:
        db.close()

    yield


app = FastAPI(
    title="SAM3 WebUI API",
    description="Backend API for SAM3 image segmentation web interface",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(images_router)
app.include_router(annotations_router)
app.include_router(processing_router)
app.include_router(jobs_router)
app.include_router(results_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}
