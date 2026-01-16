"""Tests for SAM3 inference service."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


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
    def test_process_image_calls_predict_inst(self, mock_processor_cls: MagicMock, mock_build_model: MagicMock) -> None:
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
        expected_boxes = np.array([[10, 10, 30, 30], [30, 30, 55, 55]], dtype=np.float32)
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


class TestSAM3ServiceFindAll:
    """Tests for SAM3Service find-all mode (process_image_find_all)."""

    def test_process_image_find_all_raises_when_model_not_loaded(self) -> None:
        """Test that process_image_find_all raises error when model not loaded."""
        from samui_backend.services.sam3_inference import SAM3Service

        service = SAM3Service()
        image = Image.new("RGB", (100, 100), color="red")

        with pytest.raises(RuntimeError, match="not loaded"):
            service.process_image_find_all(image, text_prompt="cat")

    def test_process_image_find_all_raises_when_no_prompts(self) -> None:
        """Test that process_image_find_all raises error when no text or boxes provided."""
        from samui_backend.services.sam3_inference import SAM3Service

        service = SAM3Service()
        service._model = MagicMock()
        image = Image.new("RGB", (100, 100), color="red")

        with pytest.raises(ValueError, match="requires text_prompt or exemplar_boxes"):
            service.process_image_find_all(image)

    @patch("samui_backend.services.sam3_inference.SAM3Service._create_transforms")
    @patch("samui_backend.services.sam3_inference.SAM3Service._create_postprocessor")
    @patch("samui_backend.services.sam3_inference.SAM3Service._create_datapoint")
    @patch("sam3.train.data.collator.collate_fn_api")
    @patch("sam3.model.utils.misc.copy_data_to_device")
    def test_process_image_find_all_with_text_prompt(
        self,
        mock_copy_to_device: MagicMock,
        mock_collate: MagicMock,
        mock_create_datapoint: MagicMock,
        mock_create_postprocessor: MagicMock,
        mock_create_transforms: MagicMock,
    ) -> None:
        """Test process_image_find_all with text prompt only."""
        import torch
        from samui_backend.services.sam3_inference import FindAllResult, SAM3Service

        # Setup mocks
        mock_model = MagicMock()
        mock_output = {"pred_logits": torch.zeros(1, 100, 256)}
        mock_model.return_value = mock_output

        mock_transform = MagicMock()
        mock_transform.return_value = MagicMock()
        mock_create_transforms.return_value = mock_transform

        mock_batch = MagicMock()
        mock_batch.find_metadatas = []
        mock_collate.return_value = {"batch": mock_batch}
        mock_copy_to_device.return_value = mock_batch

        mock_datapoint = MagicMock()
        mock_create_datapoint.return_value = mock_datapoint

        # Postprocessor returns results with masks, scores, boxes
        mock_postprocessor = MagicMock()
        mock_masks = [torch.ones(100, 100), torch.ones(100, 100)]
        mock_results = {
            1: {
                "masks": mock_masks,
                "scores": torch.tensor([0.9, 0.8]),
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 90.0, 90.0]]),
            }
        }
        mock_postprocessor.process_results.return_value = mock_results
        mock_create_postprocessor.return_value = mock_postprocessor

        # Create service and run
        service = SAM3Service()
        service._model = mock_model

        image = Image.new("RGB", (100, 100), color="red")
        result = service.process_image_find_all(image, text_prompt="cat")

        # Verify result
        assert isinstance(result, FindAllResult)
        assert result.masks.shape == (2, 100, 100)
        assert len(result.scores) == 2
        assert len(result.bboxes) == 2
        # Verify boxes converted from xyxy to xywh
        assert result.bboxes[0] == (10, 10, 40, 40)
        assert result.bboxes[1] == (60, 60, 30, 30)

        # Verify datapoint creation was called with text prompt
        mock_create_datapoint.assert_called_once_with(image, "cat", None)

    @patch("samui_backend.services.sam3_inference.SAM3Service._create_transforms")
    @patch("samui_backend.services.sam3_inference.SAM3Service._create_postprocessor")
    @patch("samui_backend.services.sam3_inference.SAM3Service._create_datapoint")
    @patch("sam3.train.data.collator.collate_fn_api")
    @patch("sam3.model.utils.misc.copy_data_to_device")
    def test_process_image_find_all_with_exemplar_boxes(
        self,
        mock_copy_to_device: MagicMock,
        mock_collate: MagicMock,
        mock_create_datapoint: MagicMock,
        mock_create_postprocessor: MagicMock,
        mock_create_transforms: MagicMock,
    ) -> None:
        """Test process_image_find_all with positive exemplar box only."""
        import torch
        from samui_backend.services.sam3_inference import FindAllResult, SAM3Service

        # Setup mocks (same as above)
        mock_model = MagicMock()
        mock_model.return_value = {}

        mock_transform = MagicMock()
        mock_transform.return_value = MagicMock()
        mock_create_transforms.return_value = mock_transform

        mock_batch = MagicMock()
        mock_batch.find_metadatas = []
        mock_collate.return_value = {"batch": mock_batch}
        mock_copy_to_device.return_value = mock_batch

        mock_datapoint = MagicMock()
        mock_create_datapoint.return_value = mock_datapoint

        mock_postprocessor = MagicMock()
        mock_masks = [torch.ones(100, 100)]
        mock_results = {
            1: {
                "masks": mock_masks,
                "scores": torch.tensor([0.95]),
                "boxes": torch.tensor([[20.0, 20.0, 80.0, 80.0]]),
            }
        }
        mock_postprocessor.process_results.return_value = mock_results
        mock_create_postprocessor.return_value = mock_postprocessor

        service = SAM3Service()
        service._model = mock_model

        image = Image.new("RGB", (100, 100), color="red")
        exemplar_boxes = [((10, 10, 30, 30), True)]  # positive exemplar
        result = service.process_image_find_all(image, exemplar_boxes=exemplar_boxes)

        # Verify result
        assert isinstance(result, FindAllResult)
        assert len(result.bboxes) == 1

        # Verify datapoint creation was called with exemplar boxes
        mock_create_datapoint.assert_called_once_with(image, None, exemplar_boxes)

    @patch("samui_backend.services.sam3_inference.SAM3Service._create_transforms")
    @patch("samui_backend.services.sam3_inference.SAM3Service._create_postprocessor")
    @patch("samui_backend.services.sam3_inference.SAM3Service._create_datapoint")
    @patch("sam3.train.data.collator.collate_fn_api")
    @patch("sam3.model.utils.misc.copy_data_to_device")
    def test_process_image_find_all_with_text_and_boxes(
        self,
        mock_copy_to_device: MagicMock,
        mock_collate: MagicMock,
        mock_create_datapoint: MagicMock,
        mock_create_postprocessor: MagicMock,
        mock_create_transforms: MagicMock,
    ) -> None:
        """Test process_image_find_all with text prompt + positive + negative boxes."""
        import torch
        from samui_backend.services.sam3_inference import FindAllResult, SAM3Service

        # Setup mocks
        mock_model = MagicMock()
        mock_model.return_value = {}

        mock_transform = MagicMock()
        mock_transform.return_value = MagicMock()
        mock_create_transforms.return_value = mock_transform

        mock_batch = MagicMock()
        mock_batch.find_metadatas = []
        mock_collate.return_value = {"batch": mock_batch}
        mock_copy_to_device.return_value = mock_batch

        mock_datapoint = MagicMock()
        mock_create_datapoint.return_value = mock_datapoint

        mock_postprocessor = MagicMock()
        mock_masks = [torch.ones(100, 100), torch.ones(100, 100), torch.ones(100, 100)]
        mock_results = {
            1: {
                "masks": mock_masks,
                "scores": torch.tensor([0.9, 0.85, 0.75]),
                "boxes": torch.tensor([
                    [10.0, 10.0, 40.0, 40.0],
                    [50.0, 50.0, 80.0, 80.0],
                    [0.0, 0.0, 20.0, 20.0],
                ]),
            }
        }
        mock_postprocessor.process_results.return_value = mock_results
        mock_create_postprocessor.return_value = mock_postprocessor

        service = SAM3Service()
        service._model = mock_model

        image = Image.new("RGB", (100, 100), color="red")
        exemplar_boxes = [
            ((10, 10, 20, 20), True),  # positive
            ((50, 50, 20, 20), False),  # negative
        ]
        result = service.process_image_find_all(image, text_prompt="person", exemplar_boxes=exemplar_boxes)

        # Verify result
        assert isinstance(result, FindAllResult)
        assert len(result.bboxes) == 3
        assert len(result.scores) == 3

        # Verify datapoint creation was called with both text and boxes
        mock_create_datapoint.assert_called_once_with(image, "person", exemplar_boxes)

    @patch("samui_backend.services.sam3_inference.SAM3Service._create_transforms")
    @patch("samui_backend.services.sam3_inference.SAM3Service._create_postprocessor")
    @patch("samui_backend.services.sam3_inference.SAM3Service._create_datapoint")
    @patch("sam3.train.data.collator.collate_fn_api")
    @patch("sam3.model.utils.misc.copy_data_to_device")
    def test_process_image_find_all_returns_empty_when_no_detections(
        self,
        mock_copy_to_device: MagicMock,
        mock_collate: MagicMock,
        mock_create_datapoint: MagicMock,
        mock_create_postprocessor: MagicMock,
        mock_create_transforms: MagicMock,
    ) -> None:
        """Test process_image_find_all returns empty result when no objects found."""
        from samui_backend.services.sam3_inference import FindAllResult, SAM3Service

        # Setup mocks
        mock_model = MagicMock()
        mock_model.return_value = {}

        mock_transform = MagicMock()
        mock_transform.return_value = MagicMock()
        mock_create_transforms.return_value = mock_transform

        mock_batch = MagicMock()
        mock_batch.find_metadatas = []
        mock_collate.return_value = {"batch": mock_batch}
        mock_copy_to_device.return_value = mock_batch

        mock_datapoint = MagicMock()
        mock_create_datapoint.return_value = mock_datapoint

        mock_postprocessor = MagicMock()
        # Empty results - no detections
        mock_postprocessor.process_results.return_value = {}
        mock_create_postprocessor.return_value = mock_postprocessor

        service = SAM3Service()
        service._model = mock_model

        image = Image.new("RGB", (100, 100), color="red")
        result = service.process_image_find_all(image, text_prompt="unicorn")

        # Verify empty result
        assert isinstance(result, FindAllResult)
        assert result.masks.shape == (0, 100, 100)
        assert len(result.scores) == 0
        assert len(result.bboxes) == 0


class TestSAM3ServicePointMode:
    """Tests for SAM3Service point mode (process_image_points)."""

    def test_process_image_points_raises_when_model_not_loaded(self) -> None:
        """Test that process_image_points raises error when model not loaded."""
        from samui_backend.services.sam3_inference import SAM3Service

        service = SAM3Service()
        image = Image.new("RGB", (100, 100), color="red")
        points = [(50, 50)]
        labels = [1]

        with pytest.raises(RuntimeError, match="not loaded"):
            service.process_image_points(image, points, labels)

    def test_process_image_points_raises_when_no_points(self) -> None:
        """Test that process_image_points raises error when no points provided."""
        from samui_backend.services.sam3_inference import SAM3Service

        service = SAM3Service()
        service._model = MagicMock()
        service._processor = MagicMock()

        image = Image.new("RGB", (100, 100), color="red")

        with pytest.raises(ValueError, match="No point annotations"):
            service.process_image_points(image, [], [])

    def test_process_image_points_raises_when_length_mismatch(self) -> None:
        """Test that process_image_points raises error when points and labels have different lengths."""
        from samui_backend.services.sam3_inference import SAM3Service

        service = SAM3Service()
        service._model = MagicMock()
        service._processor = MagicMock()

        image = Image.new("RGB", (100, 100), color="red")
        points = [(50, 50), (30, 30)]
        labels = [1]  # Only one label for two points

        with pytest.raises(ValueError, match="same length"):
            service.process_image_points(image, points, labels)

    @patch("sam3.build_sam3_image_model")
    @patch("sam3.model.sam3_image_processor.Sam3Processor")
    def test_process_image_points_calls_predict_inst_with_correct_args(
        self, mock_processor_cls: MagicMock, mock_build_model: MagicMock
    ) -> None:
        """Test that process_image_points calls predict_inst with correct point arguments."""
        from samui_backend.services.sam3_inference import SAM3Service

        # Setup mock model
        mock_model = MagicMock()
        mock_masks = np.zeros((1, 1, 100, 100), dtype=np.float32)
        mock_model.predict_inst.return_value = (mock_masks, np.array([0.95]), None)
        mock_build_model.return_value = mock_model

        # Setup mock processor
        mock_processor = MagicMock()
        mock_processor.set_image.return_value = {"state": "test"}
        mock_processor_cls.return_value = mock_processor

        service = SAM3Service()
        service.load_model()

        image = Image.new("RGB", (100, 100), color="red")
        points = [(50, 50), (30, 70)]
        labels = [1, 0]  # positive and negative

        result = service.process_image_points(image, points, labels)

        # Verify predict_inst was called with point coords and labels
        mock_model.predict_inst.assert_called_once()
        call_args = mock_model.predict_inst.call_args

        point_coords_arg = call_args.kwargs.get("point_coords")
        point_labels_arg = call_args.kwargs.get("point_labels")
        box_arg = call_args.kwargs.get("box")

        # Verify point coordinates
        expected_coords = np.array([[50, 50], [30, 70]], dtype=np.float32)
        np.testing.assert_array_equal(point_coords_arg, expected_coords)

        # Verify point labels
        expected_labels = np.array([1, 0], dtype=np.int32)
        np.testing.assert_array_equal(point_labels_arg, expected_labels)

        # Verify no bounding box
        assert box_arg is None

        # Result should have correct shape (1, H, W)
        assert result.shape == (1, 100, 100)

    @patch("sam3.build_sam3_image_model")
    @patch("sam3.model.sam3_image_processor.Sam3Processor")
    def test_process_image_points_with_positive_points_only(
        self, mock_processor_cls: MagicMock, mock_build_model: MagicMock
    ) -> None:
        """Test process_image_points with only positive points."""
        from samui_backend.services.sam3_inference import SAM3Service

        # Setup mock model
        mock_model = MagicMock()
        mock_masks = np.ones((1, 100, 100), dtype=np.float32)
        mock_model.predict_inst.return_value = (mock_masks, np.array([0.9]), None)
        mock_build_model.return_value = mock_model

        # Setup mock processor
        mock_processor = MagicMock()
        mock_processor.set_image.return_value = {"state": "test"}
        mock_processor_cls.return_value = mock_processor

        service = SAM3Service()
        service.load_model()

        image = Image.new("RGB", (100, 100), color="red")
        points = [(25, 25), (75, 75), (50, 50)]
        labels = [1, 1, 1]  # All positive

        result = service.process_image_points(image, points, labels)

        # Verify all labels are positive
        call_args = mock_model.predict_inst.call_args
        point_labels_arg = call_args.kwargs.get("point_labels")
        assert np.all(point_labels_arg == 1)

        # Result should be binary mask
        assert result.dtype == np.uint8
        assert set(np.unique(result)).issubset({0, 255})

    @patch("sam3.build_sam3_image_model")
    @patch("sam3.model.sam3_image_processor.Sam3Processor")
    def test_process_image_points_with_mixed_points(
        self, mock_processor_cls: MagicMock, mock_build_model: MagicMock
    ) -> None:
        """Test process_image_points with mixed positive and negative points."""
        from samui_backend.services.sam3_inference import SAM3Service

        # Setup mock model
        mock_model = MagicMock()
        mock_masks = np.ones((1, 100, 100), dtype=np.float32)
        mock_model.predict_inst.return_value = (mock_masks, np.array([0.85]), None)
        mock_build_model.return_value = mock_model

        # Setup mock processor
        mock_processor = MagicMock()
        mock_processor.set_image.return_value = {"state": "test"}
        mock_processor_cls.return_value = mock_processor

        service = SAM3Service()
        service.load_model()

        image = Image.new("RGB", (100, 100), color="red")
        points = [(50, 50), (10, 10), (90, 90)]
        labels = [1, 0, 1]  # positive, negative, positive

        result = service.process_image_points(image, points, labels)

        # Verify labels are correctly passed
        call_args = mock_model.predict_inst.call_args
        point_labels_arg = call_args.kwargs.get("point_labels")
        expected_labels = np.array([1, 0, 1], dtype=np.int32)
        np.testing.assert_array_equal(point_labels_arg, expected_labels)

        # Result should be single mask
        assert result.shape[0] == 1
