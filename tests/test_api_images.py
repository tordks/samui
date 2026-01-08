"""Tests for image API endpoints."""

import io
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from PIL import Image


def create_test_image() -> bytes:
    """Create a simple test image."""
    img = Image.new("RGB", (100, 100), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


class TestUploadImage:
    """Tests for POST /images endpoint."""

    def test_upload_image_success(self, client: TestClient, mock_storage: MagicMock) -> None:
        """Test successful image upload."""
        image_data = create_test_image()
        files = {"file": ("test.png", image_data, "image/png")}

        response = client.post("/images", files=files)

        assert response.status_code == 201
        data = response.json()
        assert data["filename"] == "test.png"
        assert data["width"] == 100
        assert data["height"] == 100
        assert data["processing_status"] == "pending"
        assert "id" in data

    def test_upload_invalid_file_type(self, client: TestClient) -> None:
        """Test upload with invalid file type."""
        files = {"file": ("test.txt", b"not an image", "text/plain")}

        response = client.post("/images", files=files)

        assert response.status_code == 400
        assert "image" in response.json()["detail"].lower()

    def test_upload_empty_file(self, client: TestClient) -> None:
        """Test upload with empty file."""
        files = {"file": ("test.png", b"", "image/png")}

        response = client.post("/images", files=files)

        assert response.status_code == 400


class TestListImages:
    """Tests for GET /images endpoint."""

    def test_list_images_empty(self, client: TestClient) -> None:
        """Test listing images when none exist."""
        response = client.get("/images")

        assert response.status_code == 200
        data = response.json()
        assert data["images"] == []
        assert data["total"] == 0

    def test_list_images_with_data(self, client: TestClient, mock_storage: MagicMock) -> None:
        """Test listing images after upload."""
        # Upload two images
        image_data = create_test_image()
        client.post("/images", files={"file": ("test1.png", image_data, "image/png")})
        client.post("/images", files={"file": ("test2.png", image_data, "image/png")})

        response = client.get("/images")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["images"]) == 2


class TestGetImage:
    """Tests for GET /images/{id} endpoint."""

    def test_get_image_success(self, client: TestClient, mock_storage: MagicMock) -> None:
        """Test getting a single image."""
        # Upload an image first
        image_data = create_test_image()
        upload_response = client.post("/images", files={"file": ("test.png", image_data, "image/png")})
        image_id = upload_response.json()["id"]

        response = client.get(f"/images/{image_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == image_id
        assert data["filename"] == "test.png"

    def test_get_image_not_found(self, client: TestClient) -> None:
        """Test getting non-existent image."""
        response = client.get("/images/00000000-0000-0000-0000-000000000000")

        assert response.status_code == 404


class TestDeleteImage:
    """Tests for DELETE /images/{id} endpoint."""

    def test_delete_image_success(self, client: TestClient, mock_storage: MagicMock) -> None:
        """Test deleting an image."""
        # Upload an image first
        image_data = create_test_image()
        upload_response = client.post("/images", files={"file": ("test.png", image_data, "image/png")})
        image_id = upload_response.json()["id"]

        response = client.delete(f"/images/{image_id}")

        assert response.status_code == 204

        # Verify it's deleted
        get_response = client.get(f"/images/{image_id}")
        assert get_response.status_code == 404

    def test_delete_image_not_found(self, client: TestClient) -> None:
        """Test deleting non-existent image."""
        response = client.delete("/images/00000000-0000-0000-0000-000000000000")

        assert response.status_code == 404
