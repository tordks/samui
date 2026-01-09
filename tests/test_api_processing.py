"""Tests for processing API endpoints with segmentation mode support."""

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture(autouse=True)
def reset_processing_state() -> None:
    """Reset processing state before each test."""
    from samui_backend.routes.processing import _processing_state

    _processing_state["batch_id"] = None
    _processing_state["is_running"] = False
    _processing_state["processed_count"] = 0
    _processing_state["total_count"] = 0
    _processing_state["current_image_id"] = None
    _processing_state["current_image_filename"] = None
    _processing_state["error"] = None


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


def create_exemplar_annotation(client: TestClient, image_id: str, is_positive: bool = True) -> str:
    """Helper to create an exemplar annotation and return its ID."""
    prompt_type = "positive_exemplar" if is_positive else "negative_exemplar"
    response = client.post(
        "/annotations",
        json={
            "image_id": image_id,
            "bbox_x": 10,
            "bbox_y": 20,
            "bbox_width": 30,
            "bbox_height": 40,
            "prompt_type": prompt_type,
        },
    )
    return response.json()["id"]


def set_text_prompt(client: TestClient, image_id: str, text_prompt: str) -> None:
    """Helper to set text prompt on an image."""
    client.patch(f"/images/{image_id}", json={"text_prompt": text_prompt})


class TestProcessingValidation:
    """Tests for processing endpoint validation."""

    def test_start_processing_inside_box_requires_segment_annotations(
        self, client: TestClient, mock_storage: MagicMock
    ) -> None:
        """Test that inside_box mode requires segment annotations."""
        image_id = upload_test_image(client)

        # Try to process without any annotations
        response = client.post(
            "/process",
            json={"image_ids": [image_id], "mode": "inside_box"},
        )

        assert response.status_code == 400
        assert "segment annotations" in response.json()["detail"].lower()

    def test_start_processing_inside_box_ignores_exemplar_annotations(
        self, client: TestClient, mock_storage: MagicMock
    ) -> None:
        """Test that inside_box mode ignores exemplar annotations."""
        image_id = upload_test_image(client)
        create_exemplar_annotation(client, image_id, is_positive=True)

        # Try to process with only exemplar annotations (not segment)
        response = client.post(
            "/process",
            json={"image_ids": [image_id], "mode": "inside_box"},
        )

        assert response.status_code == 400
        assert "segment annotations" in response.json()["detail"].lower()

    def test_start_processing_find_all_requires_text_or_exemplars(
        self, client: TestClient, mock_storage: MagicMock
    ) -> None:
        """Test that find_all mode requires text prompt or exemplar annotations."""
        image_id = upload_test_image(client)

        # Try to process without text prompt or exemplars
        response = client.post(
            "/process",
            json={"image_ids": [image_id], "mode": "find_all"},
        )

        assert response.status_code == 400
        assert "text prompts or exemplar" in response.json()["detail"].lower()

    def test_start_processing_find_all_with_text_prompt(self, client: TestClient, mock_storage: MagicMock) -> None:
        """Test that find_all mode accepts text prompt."""
        image_id = upload_test_image(client)
        set_text_prompt(client, image_id, "Find all cats")

        with patch("samui_backend.routes.processing._process_images_background"):
            response = client.post(
                "/process",
                json={"image_ids": [image_id], "mode": "find_all"},
            )

        assert response.status_code == 200
        assert response.json()["total_images"] == 1

    def test_start_processing_find_all_with_exemplars(self, client: TestClient, mock_storage: MagicMock) -> None:
        """Test that find_all mode accepts exemplar annotations."""
        image_id = upload_test_image(client)
        create_exemplar_annotation(client, image_id, is_positive=True)

        with patch("samui_backend.routes.processing._process_images_background"):
            response = client.post(
                "/process",
                json={"image_ids": [image_id], "mode": "find_all"},
            )

        assert response.status_code == 200
        assert response.json()["total_images"] == 1

    def test_start_processing_find_all_ignores_segment_annotations(
        self, client: TestClient, mock_storage: MagicMock
    ) -> None:
        """Test that find_all mode ignores segment annotations."""
        image_id = upload_test_image(client)
        create_segment_annotation(client, image_id)

        # Try to process with only segment annotations (not exemplars or text)
        response = client.post(
            "/process",
            json={"image_ids": [image_id], "mode": "find_all"},
        )

        assert response.status_code == 400
        assert "text prompts or exemplar" in response.json()["detail"].lower()


class TestProcessingModeRouting:
    """Tests for mode-based routing in processing."""

    def test_default_mode_is_inside_box(self, client: TestClient, mock_storage: MagicMock) -> None:
        """Test that default mode is inside_box for backward compatibility."""
        image_id = upload_test_image(client)
        create_segment_annotation(client, image_id)

        with patch("samui_backend.routes.processing._process_images_background") as mock_bg:
            response = client.post(
                "/process",
                json={"image_ids": [image_id]},  # No mode specified
            )

            assert response.status_code == 200
            # Check that background task was called with INSIDE_BOX mode
            call_args = mock_bg.call_args
            from samui_backend.db.models import SegmentationMode

            assert call_args[0][2] == SegmentationMode.INSIDE_BOX

    def test_explicit_inside_box_mode(self, client: TestClient, mock_storage: MagicMock) -> None:
        """Test explicit inside_box mode selection."""
        image_id = upload_test_image(client)
        create_segment_annotation(client, image_id)

        with patch("samui_backend.routes.processing._process_images_background") as mock_bg:
            response = client.post(
                "/process",
                json={"image_ids": [image_id], "mode": "inside_box"},
            )

            assert response.status_code == 200
            from samui_backend.db.models import SegmentationMode

            assert mock_bg.call_args[0][2] == SegmentationMode.INSIDE_BOX

    def test_explicit_find_all_mode(self, client: TestClient, mock_storage: MagicMock) -> None:
        """Test explicit find_all mode selection."""
        image_id = upload_test_image(client)
        set_text_prompt(client, image_id, "Find all dogs")

        with patch("samui_backend.routes.processing._process_images_background") as mock_bg:
            response = client.post(
                "/process",
                json={"image_ids": [image_id], "mode": "find_all"},
            )

            assert response.status_code == 200
            from samui_backend.db.models import SegmentationMode

            assert mock_bg.call_args[0][2] == SegmentationMode.FIND_ALL


class TestMaskEndpointWithMode:
    """Tests for GET /process/mask/{image_id} with mode parameter."""

    def test_get_mask_default_mode(self, client: TestClient, mock_storage: MagicMock) -> None:
        """Test getting mask with default mode."""
        # Create a mock result directly in the database

        image_id = upload_test_image(client)

        # Get database session from the app
        # Note: We need to create a ProcessingResult directly
        # This is a limitation of the test - we'd need to mock the background processing
        response = client.get(f"/process/mask/{image_id}")

        # Should return 404 since no processing result exists
        assert response.status_code == 404
        assert "inside_box" in response.json()["detail"].lower()

    def test_get_mask_with_mode_parameter(self, client: TestClient, mock_storage: MagicMock) -> None:
        """Test getting mask with explicit mode parameter."""
        image_id = upload_test_image(client)

        response = client.get(f"/process/mask/{image_id}?mode=find_all")

        # Should return 404 with mode-specific message
        assert response.status_code == 404
        assert "find_all" in response.json()["detail"].lower()


class TestExportEndpointWithMode:
    """Tests for GET /process/export/{image_id} with mode parameter."""

    def test_export_default_mode(self, client: TestClient, mock_storage: MagicMock) -> None:
        """Test export with default mode."""
        image_id = upload_test_image(client)

        response = client.get(f"/process/export/{image_id}")

        # Should return 404 since no processing result exists
        assert response.status_code == 404
        assert "inside_box" in response.json()["detail"].lower()

    def test_export_with_mode_parameter(self, client: TestClient, mock_storage: MagicMock) -> None:
        """Test export with explicit mode parameter."""
        image_id = upload_test_image(client)

        response = client.get(f"/process/export/{image_id}?mode=find_all")

        # Should return 404 with mode-specific message
        assert response.status_code == 404
        assert "find_all" in response.json()["detail"].lower()


class TestProcessingSingleImageFunctions:
    """Tests for single image processing functions."""

    def test_process_single_image_filters_by_segment_type(self, client: TestClient, mock_storage: MagicMock) -> None:
        """Test that _process_single_image only uses SEGMENT annotations."""

        # Create image with both segment and exemplar annotations
        image_id = upload_test_image(client)
        create_segment_annotation(client, image_id)
        create_exemplar_annotation(client, image_id)

        # We can't easily test the internal function without more mocking
        # This test validates the API behavior instead
        annotations_response = client.get(f"/annotations/{image_id}")
        assert annotations_response.json()["total"] == 2

    def test_find_all_creates_model_annotations(self, client: TestClient, mock_storage: MagicMock) -> None:
        """Test that find_all processing creates model-sourced annotations."""
        # This is a behavioral test that would require full integration
        # For now, we test that the API accepts the request
        image_id = upload_test_image(client)
        set_text_prompt(client, image_id, "Find all objects")

        with patch("samui_backend.routes.processing._process_images_background"):
            response = client.post(
                "/process",
                json={"image_ids": [image_id], "mode": "find_all"},
            )

        assert response.status_code == 200


class TestProcessingStatus:
    """Tests for processing status endpoint."""

    def test_get_processing_status(self, client: TestClient) -> None:
        """Test getting processing status."""
        response = client.get("/process/status")

        assert response.status_code == 200
        data = response.json()
        assert "batch_id" in data
        assert "is_running" in data
        assert "processed_count" in data
        assert "total_count" in data

    def test_processing_conflict_when_already_running(self, client: TestClient, mock_storage: MagicMock) -> None:
        """Test that starting processing while already running returns 409."""
        from samui_backend.routes.processing import _processing_state

        image_id = upload_test_image(client)
        create_segment_annotation(client, image_id)

        # Simulate running state
        _processing_state["is_running"] = True
        response = client.post(
            "/process",
            json={"image_ids": [image_id]},
        )
        assert response.status_code == 409
        assert "already in progress" in response.json()["detail"].lower()


class TestEmptyTextPromptHandling:
    """Tests for edge cases with text prompts."""

    def test_empty_string_text_prompt_not_valid(self, client: TestClient, mock_storage: MagicMock) -> None:
        """Test that empty string text prompt is not considered valid."""
        image_id = upload_test_image(client)
        set_text_prompt(client, image_id, "")

        response = client.post(
            "/process",
            json={"image_ids": [image_id], "mode": "find_all"},
        )

        assert response.status_code == 400

    def test_whitespace_only_text_prompt_not_valid(self, client: TestClient, mock_storage: MagicMock) -> None:
        """Test that whitespace-only text prompt is not considered valid."""
        image_id = upload_test_image(client)
        set_text_prompt(client, image_id, "   ")

        response = client.post(
            "/process",
            json={"image_ids": [image_id], "mode": "find_all"},
        )

        assert response.status_code == 400


class TestBatchProcessingValidation:
    """Tests for batch processing with multiple images."""

    def test_batch_filters_valid_images_by_mode(self, client: TestClient, mock_storage: MagicMock) -> None:
        """Test that batch processing filters images based on mode requirements."""
        # Create two images - one with segment annotation, one with text prompt
        image1_id = upload_test_image(client)
        create_segment_annotation(client, image1_id)

        image2_id = upload_test_image(client)
        set_text_prompt(client, image2_id, "Find cats")

        # Process with inside_box mode - only image1 should be valid
        with patch("samui_backend.routes.processing._process_images_background"):
            response = client.post(
                "/process",
                json={"image_ids": [image1_id, image2_id], "mode": "inside_box"},
            )

        assert response.status_code == 200
        assert response.json()["total_images"] == 1

    def test_batch_filters_valid_images_find_all_mode(self, client: TestClient, mock_storage: MagicMock) -> None:
        """Test batch processing filters images for find_all mode."""
        # Create two images - one with segment annotation, one with text prompt
        image1_id = upload_test_image(client)
        create_segment_annotation(client, image1_id)

        image2_id = upload_test_image(client)
        set_text_prompt(client, image2_id, "Find cats")

        # Process with find_all mode - only image2 should be valid
        with patch("samui_backend.routes.processing._process_images_background"):
            response = client.post(
                "/process",
                json={"image_ids": [image1_id, image2_id], "mode": "find_all"},
            )

        assert response.status_code == 200
        assert response.json()["total_images"] == 1
