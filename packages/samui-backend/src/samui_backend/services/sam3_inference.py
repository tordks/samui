"""SAM3 inference service for image segmentation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from samui_backend.services.sam3_batched_api import (
    boxes_xyxy_to_xywh,
    create_datapoint,
    create_postprocessor,
    create_transforms,
    normalize_mask_output,
)

logger = logging.getLogger(__name__)


@dataclass
class FindAllResult:
    """Result from find-all segmentation mode."""

    masks: NDArray[np.uint8]  # Shape: (num_objects, H, W), binary masks
    scores: NDArray[np.float32]  # Shape: (num_objects,), confidence scores
    bboxes: list[tuple[int, int, int, int]]  # List of (x, y, w, h) in pixels

    @classmethod
    def empty(cls, height: int, width: int) -> FindAllResult:
        """Create empty result for no detections."""
        return cls(
            masks=np.array([], dtype=np.uint8).reshape(0, height, width),
            scores=np.array([], dtype=np.float32),
            bboxes=[],
        )


class SAM3Service:
    """Service for running SAM3 inference on images with bounding box prompts."""

    def __init__(self) -> None:
        """Initialize the SAM3 service without loading the model."""
        self._model = None
        self._processor = None

    def load_model(self) -> None:
        """Load SAM3 model and processor into memory.

        Configures GPU settings and loads the model with instance interactivity
        enabled for box-based prompting.
        """
        if self._model is not None:
            logger.info("SAM3 model already loaded")
            return

        logger.info("Loading SAM3 model...")

        # Enable TensorFloat32 for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Import SAM3 modules
        import sam3
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        # Get BPE path for text encoder (inside sam3 package)
        bpe_path = Path(sam3.__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"

        # Build model with instance interactivity for predict_inst
        self._model = build_sam3_image_model(bpe_path=bpe_path, enable_inst_interactivity=True)
        self._processor = Sam3Processor(self._model)

        logger.info("SAM3 model loaded successfully")

    def process_image(self, image: Image.Image, bboxes: list[tuple[int, int, int, int]]) -> NDArray[np.uint8]:
        """Process an image with bounding box prompts to generate segmentation masks.

        Args:
            image: PIL Image to segment.
            bboxes: List of bounding boxes in (x, y, width, height) format (pixels).

        Returns:
            Array of binary masks with shape (num_boxes, height, width).

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._model is None or self._processor is None:
            raise RuntimeError("SAM3 model not loaded. Call load_model() first.")

        if not bboxes:
            return np.array([], dtype=np.uint8)

        # Set image in processor
        inference_state = self._processor.set_image(image)

        # Convert bboxes from xywh to xyxy format for predict_inst
        input_boxes = np.array([[x, y, x + w, y + h] for x, y, w, h in bboxes], dtype=np.float32)

        # Run inference with batched boxes
        with torch.inference_mode():
            masks, scores, _ = self._model.predict_inst(
                inference_state,
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )

        # masks shape varies: (num_boxes, num_masks, H, W) or (num_boxes, H, W)
        # Take first mask per box if 4D, then convert to binary uint8
        if masks.ndim == 4:
            masks = masks[:, 0]  # (num_boxes, H, W)
        masks = (masks > 0).astype(np.uint8) * 255

        logger.info(f"Generated {len(masks)} masks for {len(bboxes)} bboxes")
        return masks

    def process_image_points(
        self, image: Image.Image, points: list[tuple[int, int]], labels: list[int]
    ) -> NDArray[np.uint8]:
        """Process an image with point prompts to generate a segmentation mask.

        Args:
            image: PIL Image to segment.
            points: List of (x, y) point coordinates in pixels.
            labels: List of labels (1=positive/foreground, 0=negative/background).

        Returns:
            Single binary mask with shape (1, height, width).

        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If points and labels have different lengths or no points provided.
        """
        if self._model is None or self._processor is None:
            raise RuntimeError("SAM3 model not loaded. Call load_model() first.")

        if not points:
            raise ValueError("No point annotations provided.")

        if len(points) != len(labels):
            raise ValueError("Points and labels must have the same length.")

        # Set image in processor
        inference_state = self._processor.set_image(image)

        # Convert to numpy arrays
        point_coords = np.array(points, dtype=np.float32)  # shape: (N, 2)
        point_labels = np.array(labels, dtype=np.int32)  # shape: (N,)

        # Run inference with point prompts
        with torch.inference_mode():
            masks, scores, _ = self._model.predict_inst(
                inference_state,
                point_coords=point_coords,
                point_labels=point_labels,
                box=None,
                multimask_output=False,
            )

        # masks shape varies: (num_masks, H, W) or (1, num_masks, H, W)
        # Take first mask if 4D, then convert to binary uint8
        if masks.ndim == 4:
            masks = masks[0]  # (num_masks, H, W)
        if masks.ndim == 3:
            masks = masks[0:1]  # Keep as (1, H, W)
        masks = (masks > 0).astype(np.uint8) * 255

        logger.info(f"Generated mask from {len(points)} points")
        return masks

    def process_image_find_all(
        self,
        image: Image.Image,
        text_prompt: str | None = None,
        exemplar_boxes: list[tuple[tuple[int, int, int, int], bool]] | None = None,
        detection_threshold: float = 0.5,
    ) -> FindAllResult:
        """Process an image to find all instances matching text and/or exemplar boxes.

        Uses SAM3's batched API to discover all objects matching the query.

        Args:
            image: PIL Image to segment.
            text_prompt: Text description of objects to find (e.g., "person", "red car").
            exemplar_boxes: List of (bbox_xywh, is_positive) tuples providing visual examples.
                bbox_xywh is (x, y, width, height) in pixels.
                is_positive=True for positive exemplars, False for negative.
            detection_threshold: Confidence threshold for detections (0-1).

        Returns:
            FindAllResult containing masks, scores, and discovered bounding boxes.

        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If neither text_prompt nor exemplar_boxes provided.
        """
        if self._model is None:
            raise RuntimeError("SAM3 model not loaded. Call load_model() first.")

        if not text_prompt and not exemplar_boxes:
            raise ValueError("Find-all requires text_prompt or exemplar_boxes (or both).")

        from sam3.model.utils.misc import copy_data_to_device
        from sam3.train.data.collator import collate_fn_api as collate

        # Build datapoint with image and query
        datapoint = create_datapoint(image, text_prompt, exemplar_boxes)

        # Apply transforms (resize, normalize, convert boxes)
        transform = create_transforms()
        datapoint = transform(datapoint)

        # Collate into batch format and move to GPU
        batch = collate([datapoint], dict_key="batch")["batch"]
        batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)

        # Run inference
        with torch.inference_mode():
            output = self._model(batch)

        # Postprocess results
        postprocessor = create_postprocessor(detection_threshold)
        results = postprocessor.process_results(output, batch.find_metadatas)

        # Extract results (keyed by coco_image_id=1 from InferenceMetadata)
        if not results or 1 not in results:
            return FindAllResult.empty(image.height, image.width)

        result_data = results[1]
        masks = normalize_mask_output(result_data.get("masks", []), image.height, image.width)
        scores = result_data.get("scores", torch.tensor([])).float().cpu().numpy().astype(np.float32)
        bboxes = boxes_xyxy_to_xywh(result_data.get("boxes", torch.tensor([])))

        logger.info(f"Find-all discovered {len(bboxes)} objects with threshold {detection_threshold}")
        return FindAllResult(masks=masks, scores=scores, bboxes=bboxes)

    def unload_model(self) -> None:
        """Release SAM3 model from memory to free GPU resources."""
        if self._model is None:
            logger.info("SAM3 model not loaded, nothing to unload")
            return

        logger.info("Unloading SAM3 model...")

        del self._model
        del self._processor
        self._model = None
        self._processor = None

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("SAM3 model unloaded")

    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._model is not None
