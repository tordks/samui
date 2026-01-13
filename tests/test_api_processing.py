"""Tests for processing API endpoints (mask, export, status).

Note: Job creation tests are in test_api_jobs.py.
This module tests the legacy-compatible endpoints for mask/export retrieval
and the status endpoint that queries job state.
"""

import io
import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from samui_backend.db.models import ProcessingJob, ProcessingResult
from samui_backend.enums import JobStatus, SegmentationMode


def create_test_image(width: int = 100, height: int = 100) -> bytes:
    """Create a simple test image."""
    img = Image.new("RGB", (width, height), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


def upload_test_image(client: TestClient) -> str:
    """Helper to upload a test image and return its ID."""
    image_data = create_test_image()
    response = client.post("/images", files={"file": ("test.png", image_data, "image/png")})
    return response.json()["id"]


def create_segment_annotation(client: TestClient, image_id: str) -> str:
    """Helper to create a segment annotation and return its ID."""
    response = client.post(
        "/annotations",
        json={
            "image_id": image_id,
            "bbox_x": 10,
            "bbox_y": 20,
            "bbox_width": 30,
            "bbox_height": 40,
            "prompt_type": "segment",
        },
    )
    return response.json()["id"]


class TestProcessingStatus:
    """Tests for GET /process/status endpoint."""

    def test_get_processing_status_no_jobs(self, client: TestClient) -> None:
        """Test getting processing status when no jobs exist."""
        response = client.get("/process/status")

        assert response.status_code == 200
        data = response.json()
        assert data["batch_id"] is None
        assert data["is_running"] is False
        assert data["processed_count"] == 0
        assert data["total_count"] == 0
        assert data["current_image_id"] is None
        assert data["current_image_filename"] is None
        assert data["error"] is None

    def test_get_processing_status_with_running_job(
        self, client: TestClient, mock_storage: MagicMock, db_session
    ) -> None:
        """Test getting processing status when a job is running."""
        # Create an image first
        image_id = upload_test_image(client)

        # Create a running job directly in the database
        job = ProcessingJob(
            mode=SegmentationMode.INSIDE_BOX,
            status=JobStatus.RUNNING,
            image_ids=[image_id],
            current_index=0,
        )
        db_session.add(job)
        db_session.commit()

        response = client.get("/process/status")

        assert response.status_code == 200
        data = response.json()
        assert data["batch_id"] == str(job.id)
        assert data["is_running"] is True
        assert data["processed_count"] == 0
        assert data["total_count"] == 1
        assert data["current_image_id"] == image_id

    def test_get_processing_status_with_completed_job(
        self, client: TestClient, mock_storage: MagicMock, db_session
    ) -> None:
        """Test getting processing status when a job has completed."""
        # Create an image first
        image_id = upload_test_image(client)

        # Create a completed job directly in the database
        job = ProcessingJob(
            mode=SegmentationMode.INSIDE_BOX,
            status=JobStatus.COMPLETED,
            image_ids=[image_id],
            current_index=1,
        )
        db_session.add(job)
        db_session.commit()

        response = client.get("/process/status")

        assert response.status_code == 200
        data = response.json()
        assert data["batch_id"] == str(job.id)
        assert data["is_running"] is False
        assert data["processed_count"] == 1
        assert data["total_count"] == 1
        assert data["current_image_id"] is None

    def test_get_processing_status_with_failed_job(
        self, client: TestClient, mock_storage: MagicMock, db_session
    ) -> None:
        """Test getting processing status when a job has failed."""
        # Create an image first
        image_id = upload_test_image(client)

        # Create a failed job directly in the database
        job = ProcessingJob(
            mode=SegmentationMode.INSIDE_BOX,
            status=JobStatus.FAILED,
            image_ids=[image_id],
            current_index=0,
            error="Test error message",
        )
        db_session.add(job)
        db_session.commit()

        response = client.get("/process/status")

        assert response.status_code == 200
        data = response.json()
        assert data["batch_id"] == str(job.id)
        assert data["is_running"] is False
        assert data["error"] == "Test error message"


class TestMaskEndpoint:
    """Tests for GET /process/mask/{image_id} endpoint."""

    def test_get_mask_not_found(self, client: TestClient, mock_storage: MagicMock) -> None:
        """Test getting mask when no processing result exists."""
        image_id = upload_test_image(client)

        response = client.get(f"/process/mask/{image_id}")

        assert response.status_code == 404
        assert "inside_box" in response.json()["detail"].lower()

    def test_get_mask_with_mode_parameter_not_found(
        self, client: TestClient, mock_storage: MagicMock
    ) -> None:
        """Test getting mask with explicit mode parameter."""
        image_id = upload_test_image(client)

        response = client.get(f"/process/mask/{image_id}?mode=find_all")

        assert response.status_code == 404
        assert "find_all" in response.json()["detail"].lower()

    def test_get_mask_success(
        self, client: TestClient, mock_storage: MagicMock, db_session
    ) -> None:
        """Test getting mask when processing result exists."""
        # Create an image
        image_id = upload_test_image(client)

        # Create a job and result directly in the database
        job = ProcessingJob(
            mode=SegmentationMode.INSIDE_BOX,
            status=JobStatus.COMPLETED,
            image_ids=[image_id],
            current_index=1,
        )
        db_session.add(job)
        db_session.flush()

        result = ProcessingResult(
            job_id=job.id,
            image_id=uuid.UUID(image_id),
            mode=SegmentationMode.INSIDE_BOX,
            mask_blob_path="masks/test.png",
            coco_json_blob_path="coco/test.json",
        )
        db_session.add(result)
        db_session.commit()

        # Mock the storage to return mask bytes
        mask_bytes = create_test_image()
        mock_storage.get_image.return_value = mask_bytes

        response = client.get(f"/process/mask/{image_id}")

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_get_mask_returns_latest_result(
        self, client: TestClient, mock_storage: MagicMock, db_session
    ) -> None:
        """Test that mask endpoint returns the latest result."""
        # Create an image
        image_id = upload_test_image(client)

        # Create two jobs with results
        job1 = ProcessingJob(
            mode=SegmentationMode.INSIDE_BOX,
            status=JobStatus.COMPLETED,
            image_ids=[image_id],
            current_index=1,
        )
        db_session.add(job1)
        db_session.flush()

        result1 = ProcessingResult(
            job_id=job1.id,
            image_id=uuid.UUID(image_id),
            mode=SegmentationMode.INSIDE_BOX,
            mask_blob_path="masks/old.png",
            coco_json_blob_path="coco/old.json",
        )
        db_session.add(result1)
        db_session.commit()

        # Create second job with newer result
        job2 = ProcessingJob(
            mode=SegmentationMode.INSIDE_BOX,
            status=JobStatus.COMPLETED,
            image_ids=[image_id],
            current_index=1,
        )
        db_session.add(job2)
        db_session.flush()

        result2 = ProcessingResult(
            job_id=job2.id,
            image_id=uuid.UUID(image_id),
            mode=SegmentationMode.INSIDE_BOX,
            mask_blob_path="masks/new.png",
            coco_json_blob_path="coco/new.json",
        )
        db_session.add(result2)
        db_session.commit()

        # Mock the storage to return mask bytes
        mock_storage.get_image.return_value = create_test_image()

        response = client.get(f"/process/mask/{image_id}")

        assert response.status_code == 200
        # Verify the latest result's mask was requested
        mock_storage.get_image.assert_called_with("masks/new.png")


class TestExportEndpoint:
    """Tests for GET /process/export/{image_id} endpoint."""

    def test_export_not_found(self, client: TestClient, mock_storage: MagicMock) -> None:
        """Test export when no processing result exists."""
        image_id = upload_test_image(client)

        response = client.get(f"/process/export/{image_id}")

        assert response.status_code == 404
        assert "inside_box" in response.json()["detail"].lower()

    def test_export_with_mode_parameter_not_found(
        self, client: TestClient, mock_storage: MagicMock
    ) -> None:
        """Test export with explicit mode parameter."""
        image_id = upload_test_image(client)

        response = client.get(f"/process/export/{image_id}?mode=find_all")

        assert response.status_code == 404
        assert "find_all" in response.json()["detail"].lower()

    def test_export_image_not_found(self, client: TestClient) -> None:
        """Test export for non-existent image."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = client.get(f"/process/export/{fake_id}")

        assert response.status_code == 404
        assert "image not found" in response.json()["detail"].lower()

    def test_export_success(
        self, client: TestClient, mock_storage: MagicMock, db_session
    ) -> None:
        """Test export when processing result exists."""
        import json

        # Create an image
        image_id = upload_test_image(client)

        # Create a job and result directly in the database
        job = ProcessingJob(
            mode=SegmentationMode.INSIDE_BOX,
            status=JobStatus.COMPLETED,
            image_ids=[image_id],
            current_index=1,
        )
        db_session.add(job)
        db_session.flush()

        result = ProcessingResult(
            job_id=job.id,
            image_id=uuid.UUID(image_id),
            mode=SegmentationMode.INSIDE_BOX,
            mask_blob_path="masks/test.png",
            coco_json_blob_path="coco/test.json",
        )
        db_session.add(result)
        db_session.commit()

        # Mock the storage to return COCO JSON
        coco_json = {"images": [], "annotations": [], "categories": []}
        mock_storage.get_image.return_value = json.dumps(coco_json).encode("utf-8")

        response = client.get(f"/process/export/{image_id}")

        assert response.status_code == 200
        assert response.json() == coco_json
