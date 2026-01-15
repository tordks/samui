"""Tests for point annotation API endpoints."""

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


class TestCreatePointAnnotation:
    """Tests for POST /point-annotations endpoint."""

    def test_create_point_annotation_success(self, client: TestClient, mock_storage: MagicMock) -> None:
        """Test successful point annotation creation."""
        image_id = upload_test_image(client)

        response = client.post(
            "/point-annotations",
            json={
                "image_id": image_id,
                "point_x": 50,
                "point_y": 50,
                "is_positive": True,
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["image_id"] == image_id
        assert data["point_x"] == 50
        assert data["point_y"] == 50
        assert data["is_positive"] is True
        assert "id" in data
        assert "created_at" in data

    def test_create_point_annotation_negative_point(self, client: TestClient, mock_storage: MagicMock) -> None:
        """Test creating a negative point annotation."""
        image_id = upload_test_image(client)

        response = client.post(
            "/point-annotations",
            json={
                "image_id": image_id,
                "point_x": 30,
                "point_y": 40,
                "is_positive": False,
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["is_positive"] is False

    def test_create_point_annotation_default_positive(self, client: TestClient, mock_storage: MagicMock) -> None:
        """Test that is_positive defaults to True."""
        image_id = upload_test_image(client)

        response = client.post(
            "/point-annotations",
            json={
                "image_id": image_id,
                "point_x": 50,
                "point_y": 50,
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["is_positive"] is True

    def test_create_point_annotation_image_not_found(self, client: TestClient) -> None:
        """Test point annotation creation for non-existent image."""
        response = client.post(
            "/point-annotations",
            json={
                "image_id": "00000000-0000-0000-0000-000000000000",
                "point_x": 50,
                "point_y": 50,
                "is_positive": True,
            },
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_create_point_annotation_negative_coordinates(
        self, client: TestClient, mock_storage: MagicMock
    ) -> None:
        """Test point annotation creation with negative coordinates."""
        image_id = upload_test_image(client)

        response = client.post(
            "/point-annotations",
            json={
                "image_id": image_id,
                "point_x": -10,
                "point_y": 50,
                "is_positive": True,
            },
        )

        assert response.status_code == 400
        assert "non-negative" in response.json()["detail"].lower()

    def test_create_point_annotation_exceeds_width(self, client: TestClient, mock_storage: MagicMock) -> None:
        """Test point annotation creation with x coordinate exceeding image width."""
        image_id = upload_test_image(client)

        response = client.post(
            "/point-annotations",
            json={
                "image_id": image_id,
                "point_x": 100,  # Image is 100x100, valid x is 0-99
                "point_y": 50,
                "is_positive": True,
            },
        )

        assert response.status_code == 400
        assert "exceeds" in response.json()["detail"].lower()

    def test_create_point_annotation_exceeds_height(self, client: TestClient, mock_storage: MagicMock) -> None:
        """Test point annotation creation with y coordinate exceeding image height."""
        image_id = upload_test_image(client)

        response = client.post(
            "/point-annotations",
            json={
                "image_id": image_id,
                "point_x": 50,
                "point_y": 100,  # Image is 100x100, valid y is 0-99
                "is_positive": True,
            },
        )

        assert response.status_code == 400
        assert "exceeds" in response.json()["detail"].lower()

    def test_create_multiple_point_annotations_same_image(
        self, client: TestClient, mock_storage: MagicMock
    ) -> None:
        """Test creating multiple point annotations for the same image."""
        image_id = upload_test_image(client)

        response1 = client.post(
            "/point-annotations",
            json={
                "image_id": image_id,
                "point_x": 25,
                "point_y": 25,
                "is_positive": True,
            },
        )
        response2 = client.post(
            "/point-annotations",
            json={
                "image_id": image_id,
                "point_x": 75,
                "point_y": 75,
                "is_positive": False,
            },
        )

        assert response1.status_code == 201
        assert response2.status_code == 201

        # Verify both annotations exist
        get_response = client.get(f"/point-annotations/{image_id}")
        assert get_response.json()["total"] == 2


class TestGetPointAnnotations:
    """Tests for GET /point-annotations/{image_id} endpoint."""

    def test_get_point_annotations_success(self, client: TestClient, mock_storage: MagicMock) -> None:
        """Test getting point annotations for an image."""
        image_id = upload_test_image(client)

        # Create point annotation
        client.post(
            "/point-annotations",
            json={
                "image_id": image_id,
                "point_x": 50,
                "point_y": 50,
                "is_positive": True,
            },
        )

        response = client.get(f"/point-annotations/{image_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["annotations"]) == 1
        assert data["annotations"][0]["point_x"] == 50
        assert data["annotations"][0]["point_y"] == 50

    def test_get_point_annotations_empty_list(self, client: TestClient, mock_storage: MagicMock) -> None:
        """Test getting point annotations when none exist."""
        image_id = upload_test_image(client)

        response = client.get(f"/point-annotations/{image_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["annotations"] == []

    def test_get_point_annotations_image_not_found(self, client: TestClient) -> None:
        """Test getting point annotations for non-existent image."""
        response = client.get("/point-annotations/00000000-0000-0000-0000-000000000000")

        assert response.status_code == 404


class TestDeletePointAnnotation:
    """Tests for DELETE /point-annotations/{id} endpoint."""

    def test_delete_point_annotation_success(self, client: TestClient, mock_storage: MagicMock) -> None:
        """Test deleting a point annotation."""
        image_id = upload_test_image(client)

        # Create point annotation
        create_response = client.post(
            "/point-annotations",
            json={
                "image_id": image_id,
                "point_x": 50,
                "point_y": 50,
                "is_positive": True,
            },
        )
        annotation_id = create_response.json()["id"]

        # Delete annotation
        response = client.delete(f"/point-annotations/{annotation_id}")

        assert response.status_code == 204

        # Verify it's deleted
        get_response = client.get(f"/point-annotations/{image_id}")
        assert get_response.json()["total"] == 0

    def test_delete_point_annotation_not_found(self, client: TestClient) -> None:
        """Test deleting non-existent point annotation."""
        response = client.delete("/point-annotations/00000000-0000-0000-0000-000000000000")

        assert response.status_code == 404


class TestPointAnnotationsIsolation:
    """Tests for isolation between bbox and point annotations."""

    def test_bbox_and_point_annotations_are_separate(
        self, client: TestClient, mock_storage: MagicMock
    ) -> None:
        """Test that bbox annotations don't appear in point annotation list and vice versa."""
        image_id = upload_test_image(client)

        # Create bbox annotation
        client.post(
            "/annotations",
            json={
                "image_id": image_id,
                "bbox_x": 10,
                "bbox_y": 10,
                "bbox_width": 20,
                "bbox_height": 20,
            },
        )

        # Create point annotation
        client.post(
            "/point-annotations",
            json={
                "image_id": image_id,
                "point_x": 50,
                "point_y": 50,
                "is_positive": True,
            },
        )

        # Verify bbox endpoint only returns bbox annotation
        bbox_response = client.get(f"/annotations/{image_id}")
        assert bbox_response.json()["total"] == 1
        assert "bbox_x" in bbox_response.json()["annotations"][0]

        # Verify point endpoint only returns point annotation
        point_response = client.get(f"/point-annotations/{image_id}")
        assert point_response.json()["total"] == 1
        assert "point_x" in point_response.json()["annotations"][0]
