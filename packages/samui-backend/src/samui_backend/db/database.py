"""SQLAlchemy database configuration."""

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from samui_backend.config import settings

engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""

    pass


def get_db() -> Generator[Session, None, None]:
    """Dependency that provides a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_background_db() -> Generator[Session, None, None]:
    """Database session for background tasks (not request-scoped).

    Background tasks run after the HTTP response is sent, so the request-scoped
    session is closed. This context manager creates a separate session for
    background processing.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
