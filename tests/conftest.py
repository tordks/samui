"""Test fixtures for SAM3 WebUI tests."""

import os
from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

# Set test environment variables before importing app modules
os.environ["DATABASE_URL"] = "sqlite://"
os.environ["AZURE_STORAGE_CONNECTION_STRING"] = "test"
os.environ["AZURE_CONTAINER_NAME"] = "test-container"

from samui_backend.db.database import Base, get_db
from samui_backend.main import app
from samui_backend.routes.images import get_storage_service as get_storage_service_images
from samui_backend.routes.processing import get_storage_service as get_storage_service_processing


# Create test database engine (in-memory SQLite)
test_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """Create a fresh database session for each test."""
    Base.metadata.create_all(bind=test_engine)
    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def mock_storage() -> MagicMock:
    """Create a mock storage service."""
    storage_mock = MagicMock()
    storage_mock.upload_image.return_value = ("images/test.jpg", 100, 100)
    storage_mock.get_image.return_value = b"fake image data"
    storage_mock.delete_image.return_value = None
    return storage_mock


@pytest.fixture
def client(db_session: Session, mock_storage: MagicMock) -> Generator[TestClient, None, None]:
    """Create a test client with mocked dependencies."""
    def override_get_db() -> Generator[Session, None, None]:
        yield db_session

    def override_get_storage() -> MagicMock:
        return mock_storage

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_storage_service_images] = override_get_storage
    app.dependency_overrides[get_storage_service_processing] = override_get_storage

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
