"""Tests for annotation API endpoints."""

import io
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from PIL import Image


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


class TestCreateAnnotation:
    """Tests for POST /annotations endpoint."""

    def test_create_annotation_success(self, client: TestClient, mock_storage: MagicMock) -> None:
        """Test successful annotation creation."""
        image_id = upload_test_image(client)

        response = client.post(
            "/annotations",
            json={
                "image_id": image_id,
                "bbox_x": 10,
                "bbox_y": 20,
                "bbox_width": 30,
                "bbox_height": 40,
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["image_id"] == image_id
        assert data["bbox_x"] == 10
        assert data["bbox_y"] == 20
        assert data["bbox_width"] == 30
        assert data["bbox_height"] == 40
        assert "id" in data
        assert "created_at" in data

    def test_create_annotation_image_not_found(self, client: TestClient) -> None:
        """Test annotation creation for non-existent image."""
        response = client.post(
            "/annotations",
            json={
                "image_id": "00000000-0000-0000-0000-000000000000",
                "bbox_x": 10,
                "bbox_y": 20,
                "bbox_width": 30,
                "bbox_height": 40,
            },
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_create_annotation_invalid_bbox_negative_dimensions(
        self, client: TestClient, mock_storage: MagicMock
    ) -> None:
        """Test annotation creation with negative bbox dimensions."""
        image_id = upload_test_image(client)

        response = client.post(
            "/annotations",
            json={
                "image_id": image_id,
                "bbox_x": 10,
                "bbox_y": 20,
                "bbox_width": -30,
                "bbox_height": 40,
            },
        )

        assert response.status_code == 400
        assert "positive" in response.json()["detail"].lower()

    def test_create_annotation_invalid_bbox_zero_dimensions(
        self, client: TestClient, mock_storage: MagicMock
    ) -> None:
        """Test annotation creation with zero bbox dimensions."""
        image_id = upload_test_image(client)

        response = client.post(
            "/annotations",
            json={
                "image_id": image_id,
                "bbox_x": 10,
                "bbox_y": 20,
                "bbox_width": 0,
                "bbox_height": 40,
            },
        )

        assert response.status_code == 400

    def test_create_annotation_invalid_bbox_negative_coordinates(
        self, client: TestClient, mock_storage: MagicMock
    ) -> None:
        """Test annotation creation with negative bbox coordinates."""
        image_id = upload_test_image(client)

        response = client.post(
            "/annotations",
            json={
                "image_id": image_id,
                "bbox_x": -10,
                "bbox_y": 20,
                "bbox_width": 30,
                "bbox_height": 40,
            },
        )

        assert response.status_code == 400
        assert "non-negative" in response.json()["detail"].lower()

    def test_create_annotation_bbox_exceeds_image_width(
        self, client: TestClient, mock_storage: MagicMock
    ) -> None:
        """Test annotation creation with bbox exceeding image width."""
        image_id = upload_test_image(client)

        response = client.post(
            "/annotations",
            json={
                "image_id": image_id,
                "bbox_x": 80,
                "bbox_y": 20,
                "bbox_width": 30,  # 80 + 30 = 110 > 100 (image width)
                "bbox_height": 40,
            },
        )

        assert response.status_code == 400
        assert "exceeds" in response.json()["detail"].lower()

    def test_create_annotation_bbox_exceeds_image_height(
        self, client: TestClient, mock_storage: MagicMock
    ) -> None:
        """Test annotation creation with bbox exceeding image height."""
        image_id = upload_test_image(client)

        response = client.post(
            "/annotations",
            json={
                "image_id": image_id,
                "bbox_x": 10,
                "bbox_y": 80,
                "bbox_width": 30,
                "bbox_height": 30,  # 80 + 30 = 110 > 100 (image height)
            },
        )

        assert response.status_code == 400
        assert "exceeds" in response.json()["detail"].lower()

    def test_create_multiple_annotations_same_image(
        self, client: TestClient, mock_storage: MagicMock
    ) -> None:
        """Test creating multiple annotations for the same image."""
        image_id = upload_test_image(client)

        response1 = client.post(
            "/annotations",
            json={
                "image_id": image_id,
                "bbox_x": 10,
                "bbox_y": 10,
                "bbox_width": 20,
                "bbox_height": 20,
            },
        )
        response2 = client.post(
            "/annotations",
            json={
                "image_id": image_id,
                "bbox_x": 50,
                "bbox_y": 50,
                "bbox_width": 20,
                "bbox_height": 20,
            },
        )

        assert response1.status_code == 201
        assert response2.status_code == 201

        # Verify both annotations exist
        get_response = client.get(f"/annotations/{image_id}")
        assert get_response.json()["total"] == 2


class TestGetAnnotations:
    """Tests for GET /annotations/{image_id} endpoint."""

    def test_get_annotations_success(self, client: TestClient, mock_storage: MagicMock) -> None:
        """Test getting annotations for an image."""
        image_id = upload_test_image(client)

        # Create annotation
        client.post(
            "/annotations",
            json={
                "image_id": image_id,
                "bbox_x": 10,
                "bbox_y": 20,
                "bbox_width": 30,
                "bbox_height": 40,
            },
        )

        response = client.get(f"/annotations/{image_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["annotations"]) == 1
        assert data["annotations"][0]["bbox_x"] == 10

    def test_get_annotations_empty_list(
        self, client: TestClient, mock_storage: MagicMock
    ) -> None:
        """Test getting annotations when none exist."""
        image_id = upload_test_image(client)

        response = client.get(f"/annotations/{image_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["annotations"] == []

    def test_get_annotations_image_not_found(self, client: TestClient) -> None:
        """Test getting annotations for non-existent image."""
        response = client.get("/annotations/00000000-0000-0000-0000-000000000000")

        assert response.status_code == 404


class TestDeleteAnnotation:
    """Tests for DELETE /annotations/{id} endpoint."""

    def test_delete_annotation_success(
        self, client: TestClient, mock_storage: MagicMock
    ) -> None:
        """Test deleting an annotation."""
        image_id = upload_test_image(client)

        # Create annotation
        create_response = client.post(
            "/annotations",
            json={
                "image_id": image_id,
                "bbox_x": 10,
                "bbox_y": 20,
                "bbox_width": 30,
                "bbox_height": 40,
            },
        )
        annotation_id = create_response.json()["id"]

        # Delete annotation
        response = client.delete(f"/annotations/{annotation_id}")

        assert response.status_code == 204

        # Verify it's deleted
        get_response = client.get(f"/annotations/{image_id}")
        assert get_response.json()["total"] == 0

    def test_delete_annotation_not_found(self, client: TestClient) -> None:
        """Test deleting non-existent annotation."""
        response = client.delete("/annotations/00000000-0000-0000-0000-000000000000")

        assert response.status_code == 404


class TestImageStatusUpdate:
    """Tests for image status update on annotation creation."""

    def test_first_annotation_sets_status_to_annotated(
        self, client: TestClient, mock_storage: MagicMock
    ) -> None:
        """Test that first annotation changes image status from pending to annotated."""
        image_id = upload_test_image(client)

        # Verify initial status is pending
        image_response = client.get(f"/images/{image_id}")
        assert image_response.json()["processing_status"] == "pending"

        # Create annotation
        client.post(
            "/annotations",
            json={
                "image_id": image_id,
                "bbox_x": 10,
                "bbox_y": 20,
                "bbox_width": 30,
                "bbox_height": 40,
            },
        )

        # Verify status changed to annotated
        image_response = client.get(f"/images/{image_id}")
        assert image_response.json()["processing_status"] == "annotated"

    def test_second_annotation_keeps_status_annotated(
        self, client: TestClient, mock_storage: MagicMock
    ) -> None:
        """Test that subsequent annotations don't change status."""
        image_id = upload_test_image(client)

        # Create first annotation
        client.post(
            "/annotations",
            json={
                "image_id": image_id,
                "bbox_x": 10,
                "bbox_y": 20,
                "bbox_width": 30,
                "bbox_height": 40,
            },
        )

        # Create second annotation
        client.post(
            "/annotations",
            json={
                "image_id": image_id,
                "bbox_x": 50,
                "bbox_y": 50,
                "bbox_width": 20,
                "bbox_height": 20,
            },
        )

        # Verify status is still annotated
        image_response = client.get(f"/images/{image_id}")
        assert image_response.json()["processing_status"] == "annotated"
