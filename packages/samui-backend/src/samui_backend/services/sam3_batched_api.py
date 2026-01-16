"""SAM3 Batched API helpers for find-all mode.

This module provides helper functions for SAM3's Batched Inference API,
which uses DataPoints, transforms, and postprocessors. These helpers are
used by process_image_find_all in sam3_inference.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

if TYPE_CHECKING:
    from sam3.eval.postprocessors import PostProcessImage
    from sam3.train.data.sam3_image_dataset import Datapoint
    from sam3.train.transforms.basic_for_api import ComposeAPI

# Default query text when only visual exemplars are provided (no text prompt)
DEFAULT_VISUAL_QUERY = "visual"


def create_transforms() -> ComposeAPI:
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


def create_postprocessor(detection_threshold: float = 0.5) -> PostProcessImage:
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


def create_datapoint(
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
        Image as SAMImage,
        InferenceMetadata,
    )

    w, h = image.size

    # Create the image wrapper
    sam_image = SAMImage(data=image, objects=[], size=[h, w])

    # Determine query text - use default if only boxes provided
    query_text = text_prompt if text_prompt else DEFAULT_VISUAL_QUERY

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
            original_size=[h, w],  # height, width order to match SAMImage.size
            object_id=0,
            frame_index=0,
        ),
    )

    return Datapoint(images=[sam_image], find_queries=[find_query])


def normalize_mask_output(masks_data: torch.Tensor | list, height: int, width: int) -> NDArray[np.uint8]:
    """Normalize SAM3 mask output to binary numpy array.

    Args:
        masks_data: Raw mask output (tensor or list of tensors).
        height: Expected mask height.
        width: Expected mask width.

    Returns:
        Binary masks as uint8 array with shape (num_masks, H, W),
        values 0 or 255.
    """
    empty = np.array([], dtype=np.uint8).reshape(0, height, width)

    # Type dispatch: tensor vs list of tensors
    if isinstance(masks_data, torch.Tensor):
        if masks_data.numel() == 0:
            return empty
        masks = masks_data.cpu().numpy()
    elif masks_data:
        masks = torch.stack(masks_data).cpu().numpy()
    else:
        return empty

    # Shape normalization: squeeze to (num_masks, H, W)
    while masks.ndim > 3:
        masks = masks.squeeze(1)

    # Binarize
    return (masks > 0).astype(np.uint8) * 255


def boxes_xyxy_to_xywh(boxes_tensor: torch.Tensor) -> list[tuple[int, int, int, int]]:
    """Convert box tensor from xyxy to xywh format.

    Args:
        boxes_tensor: Tensor of boxes in xyxy format.

    Returns:
        List of (x, y, w, h) tuples in pixels.
    """
    if boxes_tensor.numel() == 0:
        return []
    boxes_np = boxes_tensor.float().cpu().numpy()
    return [(int(x1), int(y1), int(x2 - x1), int(y2 - y1)) for x1, y1, x2, y2 in boxes_np]
