"""SAM3 inference service for image segmentation."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

if TYPE_CHECKING:
    from sam3.eval.postprocessors import PostProcessImage
    from sam3.train.data.sam3_image_dataset import Datapoint
    from sam3.train.transforms.basic_for_api import ComposeAPI

logger = logging.getLogger(__name__)


@dataclass
class FindAllResult:
    """Result from find-all segmentation mode."""

    masks: NDArray[np.uint8]  # Shape: (num_objects, H, W), binary masks
    scores: NDArray[np.float32]  # Shape: (num_objects,), confidence scores
    bboxes: list[tuple[int, int, int, int]]  # List of (x, y, w, h) in pixels


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

    def _create_transforms(self) -> ComposeAPI:
        """Create transform pipeline for batched API inference.

        Returns:
            Configured transform pipeline that resizes to 1008x1008 and normalizes.
        """
        from sam3.train.transforms.basic_for_api import (
            ComposeAPI,
            NormalizeAPI,
            RandomResizeAPI,
            ToTensorAPI,
        )

        return ComposeAPI(
            transforms=[
                RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _create_postprocessor(self, detection_threshold: float = 0.5) -> PostProcessImage:
        """Create postprocessor for batched API results.

        Args:
            detection_threshold: Confidence threshold for detections.

        Returns:
            Configured postprocessor that outputs masks at original image size.
        """
        from sam3.eval.postprocessors import PostProcessImage

        return PostProcessImage(
            max_dets_per_img=-1,
            iou_type="segm",
            use_original_sizes_box=True,
            use_original_sizes_mask=True,
            convert_mask_to_rle=False,
            detection_threshold=detection_threshold,
            to_cpu=False,
        )

    def _create_datapoint(
        self,
        image: Image.Image,
        text_prompt: str | None,
        exemplar_boxes: list[tuple[tuple[int, int, int, int], bool]] | None,
    ) -> Datapoint:
        """Create a DataPoint for batched API inference.

        Args:
            image: PIL Image to process.
            text_prompt: Text query for finding objects (e.g., "cat", "person").
            exemplar_boxes: List of (bbox_xywh, is_positive) tuples. Bbox format is (x, y, w, h) in pixels.

        Returns:
            Configured DataPoint ready for transforms and inference.
        """
        from sam3.train.data.sam3_image_dataset import (
            Datapoint,
            FindQueryLoaded,
            InferenceMetadata,
        )
        from sam3.train.data.sam3_image_dataset import (
            Image as SAMImage,
        )

        w, h = image.size

        # Create the image wrapper
        sam_image = SAMImage(data=image, objects=[], size=[h, w])

        # Determine query text - use "visual" if only boxes provided
        query_text = text_prompt if text_prompt else "visual"

        # Build box tensors if exemplars provided
        input_bbox = None
        input_bbox_label = None
        if exemplar_boxes:
            # Convert xywh to xyxy format for the batched API
            boxes_xyxy = []
            labels = []
            for (x, y, bw, bh), is_positive in exemplar_boxes:
                boxes_xyxy.append([x, y, x + bw, y + bh])
                labels.append(is_positive)

            input_bbox = torch.tensor(boxes_xyxy, dtype=torch.float).view(-1, 4)
            input_bbox_label = torch.tensor(labels, dtype=torch.bool).view(-1)

        # Create query with metadata
        find_query = FindQueryLoaded(
            query_text=query_text,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            input_bbox=input_bbox,
            input_bbox_label=input_bbox_label,
            inference_metadata=InferenceMetadata(
                coco_image_id=1,
                original_image_id=1,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            ),
        )

        return Datapoint(images=[sam_image], find_queries=[find_query])

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
        datapoint = self._create_datapoint(image, text_prompt, exemplar_boxes)

        # Apply transforms (resize, normalize, convert boxes)
        transform = self._create_transforms()
        datapoint = transform(datapoint)

        # Collate into batch format and move to GPU
        batch = collate([datapoint], dict_key="batch")["batch"]
        batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)

        # Run inference
        with torch.inference_mode():
            output = self._model(batch)

        # Postprocess results
        postprocessor = self._create_postprocessor(detection_threshold)
        results = postprocessor.process_results(output, batch.find_metadatas)

        # Extract masks, scores, and boxes from results
        # Results are keyed by coco_image_id from InferenceMetadata
        if not results or 1 not in results:
            # No detections found
            return FindAllResult(
                masks=np.array([], dtype=np.uint8).reshape(0, image.height, image.width),
                scores=np.array([], dtype=np.float32),
                bboxes=[],
            )

        result_data = results[1]
        masks_list = result_data.get("masks", [])
        scores_tensor = result_data.get("scores", torch.tensor([]))
        boxes_tensor = result_data.get("boxes", torch.tensor([]))

        # Convert masks to numpy binary format
        if masks_list:
            # Masks are returned as list of tensors
            masks = torch.stack(masks_list).cpu().numpy()
            masks = (masks > 0).astype(np.uint8) * 255
        else:
            masks = np.array([], dtype=np.uint8).reshape(0, image.height, image.width)

        # Convert scores to numpy
        scores = scores_tensor.cpu().numpy().astype(np.float32)

        # Convert boxes from xyxy to xywh format
        bboxes = []
        if boxes_tensor.numel() > 0:
            boxes_np = boxes_tensor.cpu().numpy()
            for x1, y1, x2, y2 in boxes_np:
                bboxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))

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
