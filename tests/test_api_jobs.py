"""Tests for job API endpoints."""

import io
import uuid
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from PIL import Image as PILImage
from sqlalchemy.orm import Session

from samui_backend.db.models import Annotation, Image, ProcessingJob, ProcessingResult
from samui_backend.enums import JobStatus, PromptType, SegmentationMode


def create_test_image_bytes() -> bytes:
    """Create a simple test image."""
    img = PILImage.new("RGB", (100, 100), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


def create_test_image(db: Session, filename: str = "test.jpg") -> Image:
    """Create a test image in the database."""
    image = Image(
        filename=filename,
        blob_path=f"images/{filename}",
        width=100,
        height=100,
    )
    db.add(image)
    db.commit()
    return image


def create_test_annotation(
    db: Session,
    image_id: uuid.UUID,
    prompt_type: PromptType = PromptType.SEGMENT,
) -> Annotation:
    """Create a test annotation in the database."""
    annotation = Annotation(
        image_id=image_id,
        bbox_x=10,
        bbox_y=10,
        bbox_width=50,
        bbox_height=50,
        prompt_type=prompt_type,
    )
    db.add(annotation)
    db.commit()
    return annotation


def create_test_job(
    db: Session,
    image_ids: list[uuid.UUID],
    mode: SegmentationMode = SegmentationMode.INSIDE_BOX,
    status: JobStatus = JobStatus.QUEUED,
    filenames: list[str] | None = None,
) -> ProcessingJob:
    """Create a test processing job."""
    job = ProcessingJob(
        mode=mode,
        status=status,
        image_ids=[str(img_id) for img_id in image_ids],
        image_filenames=filenames or [f"test_{i}.png" for i in range(len(image_ids))],
    )
    db.add(job)
    db.commit()
    return job


def create_test_result(
    db: Session,
    job_id: uuid.UUID,
    image_id: uuid.UUID,
    mode: SegmentationMode = SegmentationMode.INSIDE_BOX,
    annotation_ids: list[str] | None = None,
    text_prompt_used: str | None = None,
) -> ProcessingResult:
    """Create a test processing result."""
    result = ProcessingResult(
        job_id=job_id,
        image_id=image_id,
        mode=mode,
        mask_blob_path=f"masks/{uuid.uuid4()}.png",
        coco_json_blob_path=f"coco/{uuid.uuid4()}.json",
        annotation_ids=annotation_ids,
        text_prompt_used=text_prompt_used,
    )
    db.add(result)
    db.commit()
    return result


class TestCreateJob:
    """Tests for POST /jobs endpoint."""

    @patch("samui_backend.routes.jobs.start_job_if_none_running")
    def test_create_job_success(
        self, mock_start: MagicMock, client: TestClient, db_session: Session
    ) -> None:
        """Test creating a job with valid images."""
        image = create_test_image(db_session)
        create_test_annotation(db_session, image.id)

        response = client.post(
            "/jobs",
            json={
                "image_ids": [str(image.id)],
                "mode": "inside_box",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "queued"
        assert data["mode"] == "inside_box"
        assert len(data["image_ids"]) == 1
        assert data["image_count"] == 1
        mock_start.assert_called_once()

    @patch("samui_backend.routes.jobs.start_job_if_none_running")
    def test_create_job_with_force_all(
        self, mock_start: MagicMock, client: TestClient, db_session: Session
    ) -> None:
        """Test creating a job with force_all=True includes all images."""
        # Create image with annotation and existing result
        image = create_test_image(db_session)
        ann = create_test_annotation(db_session, image.id)
        job = create_test_job(db_session, [image.id])
        create_test_result(db_session, job.id, image.id, annotation_ids=[str(ann.id)])

        # Without force_all, no images need processing
        response = client.post(
            "/jobs",
            json={
                "image_ids": [str(image.id)],
                "mode": "inside_box",
                "force_all": False,
            },
        )
        assert response.status_code == 400
        assert "No images need processing" in response.json()["detail"]

        # With force_all, image is included
        response = client.post(
            "/jobs",
            json={
                "image_ids": [str(image.id)],
                "mode": "inside_box",
                "force_all": True,
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert len(data["image_ids"]) == 1

    def test_create_job_no_valid_images(self, client: TestClient) -> None:
        """Test creating a job with non-existent image IDs."""
        response = client.post(
            "/jobs",
            json={
                "image_ids": [str(uuid.uuid4())],
                "mode": "inside_box",
            },
        )

        assert response.status_code == 400
        assert "No valid image IDs" in response.json()["detail"]

    def test_create_job_no_images_need_processing(self, client: TestClient, db_session: Session) -> None:
        """Test creating a job when no images need processing."""
        # Image without annotations doesn't need processing
        image = create_test_image(db_session)

        response = client.post(
            "/jobs",
            json={
                "image_ids": [str(image.id)],
                "mode": "inside_box",
            },
        )

        assert response.status_code == 400
        assert "No images need processing" in response.json()["detail"]

    @patch("samui_backend.routes.jobs.start_job_if_none_running")
    def test_create_job_find_all_mode(
        self, mock_start: MagicMock, client: TestClient, db_session: Session
    ) -> None:
        """Test creating a find-all mode job."""
        image = create_test_image(db_session)
        image.text_prompt = "find all cats"
        db_session.commit()

        response = client.post(
            "/jobs",
            json={
                "image_ids": [str(image.id)],
                "mode": "find_all",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["mode"] == "find_all"


class TestListJobs:
    """Tests for GET /jobs endpoint."""

    def test_list_jobs_empty(self, client: TestClient) -> None:
        """Test listing jobs when none exist."""
        response = client.get("/jobs")

        assert response.status_code == 200
        assert response.json() == []

    def test_list_jobs_with_data(self, client: TestClient, db_session: Session) -> None:
        """Test listing jobs returns newest first."""
        img1 = create_test_image(db_session, "test1.jpg")
        img2 = create_test_image(db_session, "test2.jpg")
        job1 = create_test_job(db_session, [img1.id])
        job2 = create_test_job(db_session, [img2.id])

        response = client.get("/jobs")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        # Newest first
        assert data[0]["id"] == str(job2.id)
        assert data[1]["id"] == str(job1.id)


class TestGetJob:
    """Tests for GET /jobs/{job_id} endpoint."""

    def test_get_job_success(self, client: TestClient, db_session: Session) -> None:
        """Test getting a job by ID."""
        image = create_test_image(db_session)
        job = create_test_job(db_session, [image.id])

        response = client.get(f"/jobs/{job.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(job.id)
        assert data["status"] == "queued"
        assert data["current_index"] == 0

    def test_get_job_not_found(self, client: TestClient) -> None:
        """Test getting non-existent job."""
        response = client.get(f"/jobs/{uuid.uuid4()}")

        assert response.status_code == 404


class TestGetImageHistory:
    """Tests for GET /images/{image_id}/history endpoint."""

    def test_get_history_empty(self, client: TestClient, db_session: Session) -> None:
        """Test getting history when no results exist."""
        image = create_test_image(db_session)

        response = client.get(f"/images/{image.id}/history")

        assert response.status_code == 200
        assert response.json() == []

    def test_get_history_with_results(self, client: TestClient, db_session: Session) -> None:
        """Test getting history with multiple results."""
        image = create_test_image(db_session)
        job = create_test_job(db_session, [image.id])
        result1 = create_test_result(db_session, job.id, image.id)
        result2 = create_test_result(db_session, job.id, image.id)

        response = client.get(f"/images/{image.id}/history")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        # Newest first
        assert data[0]["id"] == str(result2.id)
        assert data[1]["id"] == str(result1.id)

    def test_get_history_filters_by_mode(self, client: TestClient, db_session: Session) -> None:
        """Test history is filtered by mode."""
        image = create_test_image(db_session)
        job = create_test_job(db_session, [image.id])
        create_test_result(db_session, job.id, image.id, mode=SegmentationMode.INSIDE_BOX)
        create_test_result(db_session, job.id, image.id, mode=SegmentationMode.FIND_ALL)

        # Default mode is inside_box
        response = client.get(f"/images/{image.id}/history")
        assert len(response.json()) == 1

        # Explicit find_all mode
        response = client.get(f"/images/{image.id}/history?mode=find_all")
        assert len(response.json()) == 1

    def test_get_history_image_not_found(self, client: TestClient) -> None:
        """Test getting history for non-existent image."""
        response = client.get(f"/images/{uuid.uuid4()}/history")

        assert response.status_code == 404


class TestGetResultMask:
    """Tests for GET /results/{result_id}/mask endpoint."""

    def test_get_mask_success(self, client: TestClient, db_session: Session, mock_storage: MagicMock) -> None:
        """Test getting mask for a result."""
        image = create_test_image(db_session)
        job = create_test_job(db_session, [image.id])
        result = create_test_result(db_session, job.id, image.id)

        response = client.get(f"/results/{result.id}/mask")

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        mock_storage.get_image.assert_called_with(result.mask_blob_path)

    def test_get_mask_not_found(self, client: TestClient) -> None:
        """Test getting mask for non-existent result."""
        response = client.get(f"/results/{uuid.uuid4()}/mask")

        assert response.status_code == 404

    def test_get_mask_no_mask_available(self, client: TestClient, db_session: Session) -> None:
        """Test getting mask when result has no mask."""
        image = create_test_image(db_session)
        job = create_test_job(db_session, [image.id])
        result = ProcessingResult(
            job_id=job.id,
            image_id=image.id,
            mode=SegmentationMode.INSIDE_BOX,
            mask_blob_path="",  # Empty path
            coco_json_blob_path="coco/test.json",
        )
        db_session.add(result)
        db_session.commit()

        response = client.get(f"/results/{result.id}/mask")

        assert response.status_code == 404
        assert "not available" in response.json()["detail"]
