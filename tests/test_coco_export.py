"""Tests for COCO export service."""

import uuid

import numpy as np
import pytest


class TestMaskToRLE:
    """Tests for _mask_to_rle function."""

    def test_empty_mask_returns_zeros(self) -> None:
        """Test that empty mask returns all zeros run."""
        from samui_backend.services.coco_export import _mask_to_rle

        mask = np.zeros((10, 10), dtype=np.uint8)
        result = _mask_to_rle(mask)

        assert "counts" in result
        assert "size" in result
        assert result["size"] == [10, 10]
        # Single run of 100 zeros
        assert result["counts"] == [100]

    def test_full_mask_returns_single_run(self) -> None:
        """Test that full mask returns run with prepended zero for ones-start."""
        from samui_backend.services.coco_export import _mask_to_rle

        mask = np.full((10, 10), 255, dtype=np.uint8)
        result = _mask_to_rle(mask)

        assert result["size"] == [10, 10]
        # COCO RLE starts with zeros count; if mask starts with 1, prepend 0
        # First run: 0 zeros, then 100 ones
        assert len(result["counts"]) >= 1
        # First element should be 0 (no zeros before the ones)
        assert result["counts"][0] == 0
        # Total should sum to 100 pixels
        assert sum(result["counts"]) == 100

    def test_alternating_mask(self) -> None:
        """Test RLE with simple alternating pattern."""
        from samui_backend.services.coco_export import _mask_to_rle

        # Create 2x2 mask with known pattern
        mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        result = _mask_to_rle(mask)

        assert result["size"] == [2, 2]
        # Column-major order: [0,255] then [255,0] -> 0,1,1,0
        # Runs: 1 zero, 2 ones, 1 zero = [1, 2, 1]
        assert result["counts"] == [1, 2, 1]


class TestComputeBboxArea:
    """Tests for _compute_bbox_area function."""

    def test_empty_mask_returns_zero(self) -> None:
        """Test that empty mask returns zero area."""
        from samui_backend.services.coco_export import _compute_bbox_area

        mask = np.zeros((100, 100), dtype=np.uint8)
        assert _compute_bbox_area(mask) == 0

    def test_full_mask_returns_total_pixels(self) -> None:
        """Test that full mask returns total pixel count."""
        from samui_backend.services.coco_export import _compute_bbox_area

        mask = np.full((100, 100), 255, dtype=np.uint8)
        assert _compute_bbox_area(mask) == 10000

    def test_partial_mask_returns_correct_count(self) -> None:
        """Test that partial mask returns correct pixel count."""
        from samui_backend.services.coco_export import _compute_bbox_area

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:50, 0:50] = 255  # 2500 pixels
        assert _compute_bbox_area(mask) == 2500


class TestGenerateCocoJson:
    """Tests for generate_coco_json function."""

    def test_generates_valid_coco_structure(self) -> None:
        """Test that output has valid COCO structure."""
        from samui_backend.services.coco_export import generate_coco_json

        image_id = uuid.uuid4()
        masks = np.zeros((2, 100, 100), dtype=np.uint8)
        masks[0, 10:30, 10:30] = 255
        masks[1, 50:70, 50:70] = 255

        result = generate_coco_json(
            image_id=image_id,
            filename="test.jpg",
            width=100,
            height=100,
            bboxes=[(10, 10, 20, 20), (50, 50, 20, 20)],
            masks=masks,
        )

        # Check required top-level keys
        assert "images" in result
        assert "annotations" in result
        assert "categories" in result

        # Check images array
        assert len(result["images"]) == 1
        image_info = result["images"][0]
        assert image_info["id"] == str(image_id)
        assert image_info["file_name"] == "test.jpg"
        assert image_info["width"] == 100
        assert image_info["height"] == 100

        # Check categories array
        assert len(result["categories"]) == 1
        assert result["categories"][0]["id"] == 1
        assert result["categories"][0]["name"] == "object"

        # Check annotations array
        assert len(result["annotations"]) == 2

    def test_annotation_has_required_fields(self) -> None:
        """Test that each annotation has required COCO fields."""
        from samui_backend.services.coco_export import generate_coco_json

        image_id = uuid.uuid4()
        masks = np.zeros((1, 100, 100), dtype=np.uint8)
        masks[0, 10:30, 10:30] = 255

        result = generate_coco_json(
            image_id=image_id,
            filename="test.jpg",
            width=100,
            height=100,
            bboxes=[(10, 10, 20, 20)],
            masks=masks,
        )

        annotation = result["annotations"][0]

        # Required COCO annotation fields
        assert "id" in annotation
        assert "image_id" in annotation
        assert "category_id" in annotation
        assert "segmentation" in annotation
        assert "bbox" in annotation
        assert "area" in annotation
        assert "iscrowd" in annotation

        # Check field values
        assert annotation["id"] == 1
        assert annotation["image_id"] == str(image_id)
        assert annotation["category_id"] == 1
        assert annotation["bbox"] == [10, 10, 20, 20]
        assert annotation["iscrowd"] == 0

    def test_segmentation_has_rle_format(self) -> None:
        """Test that segmentation is in RLE format."""
        from samui_backend.services.coco_export import generate_coco_json

        image_id = uuid.uuid4()
        masks = np.zeros((1, 100, 100), dtype=np.uint8)
        masks[0, 10:30, 10:30] = 255

        result = generate_coco_json(
            image_id=image_id,
            filename="test.jpg",
            width=100,
            height=100,
            bboxes=[(10, 10, 20, 20)],
            masks=masks,
        )

        segmentation = result["annotations"][0]["segmentation"]

        # RLE format has counts and size
        assert "counts" in segmentation
        assert "size" in segmentation
        assert isinstance(segmentation["counts"], list)
        assert segmentation["size"] == [100, 100]

    def test_handles_empty_annotations(self) -> None:
        """Test that empty bboxes/masks produces empty annotations."""
        from samui_backend.services.coco_export import generate_coco_json

        image_id = uuid.uuid4()
        masks = np.zeros((0, 100, 100), dtype=np.uint8)

        result = generate_coco_json(
            image_id=image_id,
            filename="test.jpg",
            width=100,
            height=100,
            bboxes=[],
            masks=masks,
        )

        assert len(result["annotations"]) == 0
        assert len(result["images"]) == 1

    def test_custom_category_name(self) -> None:
        """Test that custom category name is used."""
        from samui_backend.services.coco_export import generate_coco_json

        image_id = uuid.uuid4()
        masks = np.zeros((1, 100, 100), dtype=np.uint8)

        result = generate_coco_json(
            image_id=image_id,
            filename="test.jpg",
            width=100,
            height=100,
            bboxes=[(0, 0, 10, 10)],
            masks=masks,
            category_name="person",
        )

        assert result["categories"][0]["name"] == "person"
