"""Tests for SAM3 inference service."""

import numpy as np
import pytest
from PIL import Image
from unittest.mock import MagicMock, patch


class TestSAM3Service:
    """Tests for SAM3Service class."""

    def test_service_initializes_unloaded(self) -> None:
        """Test that service initializes without loading model."""
        from samui_backend.services.sam3_inference import SAM3Service

        service = SAM3Service()
        assert not service.is_loaded

    def test_process_image_raises_when_model_not_loaded(self) -> None:
        """Test that process_image raises error when model not loaded."""
        from samui_backend.services.sam3_inference import SAM3Service

        service = SAM3Service()
        image = Image.new("RGB", (100, 100), color="red")
        bboxes = [(10, 10, 20, 20)]

        with pytest.raises(RuntimeError, match="not loaded"):
            service.process_image(image, bboxes)

    def test_process_image_returns_empty_array_for_empty_bboxes(self) -> None:
        """Test that process_image returns empty array when no bboxes provided."""
        from samui_backend.services.sam3_inference import SAM3Service

        service = SAM3Service()
        # Mock the model as loaded
        service._model = MagicMock()
        service._processor = MagicMock()

        image = Image.new("RGB", (100, 100), color="red")
        result = service.process_image(image, [])

        assert len(result) == 0

    @patch("sam3.build_sam3_image_model")
    @patch("sam3.model.sam3_image_processor.Sam3Processor")
    def test_load_model_initializes_model_and_processor(
        self, mock_processor_cls: MagicMock, mock_build_model: MagicMock
    ) -> None:
        """Test that load_model initializes model and processor."""
        from samui_backend.services.sam3_inference import SAM3Service

        mock_model = MagicMock()
        mock_build_model.return_value = mock_model

        mock_processor = MagicMock()
        mock_processor_cls.return_value = mock_processor

        service = SAM3Service()
        service.load_model()

        assert service.is_loaded
        mock_build_model.assert_called_once()
        mock_processor_cls.assert_called_once_with(mock_model)

    @patch("sam3.build_sam3_image_model")
    @patch("sam3.model.sam3_image_processor.Sam3Processor")
    def test_process_image_calls_predict_inst(
        self, mock_processor_cls: MagicMock, mock_build_model: MagicMock
    ) -> None:
        """Test that process_image calls predict_inst with correct arguments."""
        from samui_backend.services.sam3_inference import SAM3Service

        # Setup mock model
        mock_model = MagicMock()
        mock_masks = np.zeros((2, 1, 100, 100), dtype=np.float32)
        mock_model.predict_inst.return_value = (mock_masks, np.array([0.9, 0.8]), None)
        mock_build_model.return_value = mock_model

        # Setup mock processor
        mock_processor = MagicMock()
        mock_processor.set_image.return_value = {"state": "test"}
        mock_processor_cls.return_value = mock_processor

        service = SAM3Service()
        service.load_model()

        image = Image.new("RGB", (100, 100), color="red")
        bboxes = [(10, 10, 20, 20), (30, 30, 25, 25)]

        result = service.process_image(image, bboxes)

        # Verify predict_inst was called with xyxy format boxes
        mock_model.predict_inst.assert_called_once()
        call_args = mock_model.predict_inst.call_args
        boxes_arg = call_args.kwargs.get("box")
        if boxes_arg is None:
            boxes_arg = call_args[1].get("box")

        # Bboxes should be converted from xywh to xyxy
        expected_boxes = np.array(
            [[10, 10, 30, 30], [30, 30, 55, 55]], dtype=np.float32
        )
        np.testing.assert_array_equal(boxes_arg, expected_boxes)

        # Result should have correct shape
        assert result.shape == (2, 100, 100)

    def test_unload_model_clears_references(self) -> None:
        """Test that unload_model clears model and processor references."""
        from samui_backend.services.sam3_inference import SAM3Service

        service = SAM3Service()
        service._model = MagicMock()
        service._processor = MagicMock()

        assert service.is_loaded

        service.unload_model()

        assert not service.is_loaded
        assert service._model is None
        assert service._processor is None

    def test_unload_model_is_safe_when_not_loaded(self) -> None:
        """Test that unload_model doesn't raise when model not loaded."""
        from samui_backend.services.sam3_inference import SAM3Service

        service = SAM3Service()
        # Should not raise
        service.unload_model()
        assert not service.is_loaded
