"""SAM3 inference service for image segmentation."""

import logging
import os

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

logger = logging.getLogger(__name__)


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
        bpe_path = os.path.join(os.path.dirname(sam3.__file__), "assets", "bpe_simple_vocab_16e6.txt.gz")

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
