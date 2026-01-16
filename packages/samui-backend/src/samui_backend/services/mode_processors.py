"""Mode-specific image processing functions for SAM3 inference.

This module contains the processing logic for each segmentation mode:
- INSIDE_BOX: Process bounding box prompts
- FIND_ALL: Process text prompts and exemplar boxes
- POINT: Process point prompts
"""

from __future__ import annotations

import json
import logging
import uuid
from io import BytesIO

import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage

from samui_backend.db.models import Image, ProcessingResult
from samui_backend.enums import PromptType
from samui_backend.schemas import BboxAnnotationSnapshot, PointAnnotationSnapshot
from samui_backend.services.coco_export import generate_coco_json
from samui_backend.services.sam3_inference import SAM3Service
from samui_backend.services.storage import StorageService

logger = logging.getLogger(__name__)


def save_mask_to_storage(
    storage: StorageService,
    masks: NDArray[np.uint8],
    result_id: uuid.UUID,
) -> str:
    """Save combined mask image to storage using result_id for history support."""
    mask_blob_path = f"masks/{result_id}.png"

    if masks.size == 0:
        # Empty mask for no detections
        combined_mask = np.zeros((1, 1), dtype=np.uint8)
    else:
        combined_mask = np.zeros((masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for mask in masks:
            combined_mask = np.maximum(combined_mask, mask)

    mask_image = PILImage.fromarray(combined_mask)
    mask_buffer = BytesIO()
    mask_image.save(mask_buffer, format="PNG")
    mask_bytes = mask_buffer.getvalue()

    storage.upload_blob(mask_blob_path, mask_bytes, content_type="image/png")
    return mask_blob_path


def save_coco_to_storage(
    storage: StorageService,
    image: Image,
    bboxes: list[tuple[int, int, int, int]],
    masks: NDArray[np.uint8],
    result_id: uuid.UUID,
    points: list[tuple[int, int, bool]] | None = None,
) -> str:
    """Generate and save COCO JSON to storage using result_id for history support."""
    coco_blob_path = f"coco/{result_id}.json"
    coco_json = generate_coco_json(
        image_id=image.id,
        filename=image.filename,
        width=image.width,
        height=image.height,
        bboxes=bboxes,
        masks=masks,
        points=points,
    )
    coco_bytes = json.dumps(coco_json, indent=2).encode("utf-8")
    storage.upload_blob(coco_blob_path, coco_bytes, content_type="application/json")
    return coco_blob_path


def process_inside_box(
    storage: StorageService,
    sam3: SAM3Service,
    image: Image,
    pil_image: PILImage.Image,
    bbox_annotations: list[BboxAnnotationSnapshot],
    result: ProcessingResult,
) -> bool:
    """Process image with bounding box prompts (INSIDE_BOX mode).

    Returns:
        True if successful, False if no annotations to process.
    """
    if not bbox_annotations:
        logger.warning(f"No segment annotations for image {image.id}, skipping")
        return False

    bboxes = [(ann.bbox_x, ann.bbox_y, ann.bbox_width, ann.bbox_height) for ann in bbox_annotations]
    masks = sam3.process_image(pil_image, bboxes)

    result.mask_blob_path = save_mask_to_storage(storage, masks, result.id)
    result.coco_json_blob_path = save_coco_to_storage(storage, image, bboxes, masks, result.id)
    return True


def process_find_all(
    storage: StorageService,
    sam3: SAM3Service,
    image: Image,
    pil_image: PILImage.Image,
    bbox_annotations: list[BboxAnnotationSnapshot],
    text_prompt: str | None,
    result: ProcessingResult,
) -> bool:
    """Process image with text prompt and/or exemplar boxes (FIND_ALL mode).

    Returns:
        True if successful, False if no text prompt or exemplars to process.
    """
    if not text_prompt and not bbox_annotations:
        logger.warning(f"No text prompt or exemplars for image {image.id} in find-all mode, skipping")
        return False

    # Convert exemplar annotations to (bbox_xywh, is_positive) format
    exemplar_boxes = None
    if bbox_annotations:
        exemplar_boxes = [
            (
                (ann.bbox_x, ann.bbox_y, ann.bbox_width, ann.bbox_height),
                ann.prompt_type == PromptType.POSITIVE_EXEMPLAR,
            )
            for ann in bbox_annotations
        ]

    # Run find-all inference
    find_result = sam3.process_image_find_all(pil_image, text_prompt, exemplar_boxes)

    # Store discovered bboxes in result
    result.bboxes = [{"x": x, "y": y, "width": w, "height": h} for x, y, w, h in find_result.bboxes]

    # Save masks and COCO JSON
    if find_result.masks.size > 0:
        result.mask_blob_path = save_mask_to_storage(storage, find_result.masks, result.id)
        result.coco_json_blob_path = save_coco_to_storage(
            storage, image, find_result.bboxes, find_result.masks, result.id
        )
    else:
        # No discoveries - save empty results
        empty_masks = np.zeros((0, pil_image.height, pil_image.width), dtype=np.uint8)
        result.mask_blob_path = save_mask_to_storage(storage, empty_masks, result.id)
        result.coco_json_blob_path = save_coco_to_storage(storage, image, [], empty_masks, result.id)

    logger.info(f"Find-all discovered {len(find_result.bboxes)} objects for image {image.id}")
    return True


def process_point(
    storage: StorageService,
    sam3: SAM3Service,
    image: Image,
    pil_image: PILImage.Image,
    point_annotations: list[PointAnnotationSnapshot],
    result: ProcessingResult,
) -> bool:
    """Process image with point prompts (POINT mode).

    Returns:
        True if successful, False if no point annotations to process.
    """
    if not point_annotations:
        logger.warning(f"No point annotations for image {image.id}, skipping")
        return False

    # Extract coordinates and labels from point annotations
    points = [(ann.point_x, ann.point_y) for ann in point_annotations]
    labels = [1 if ann.is_positive else 0 for ann in point_annotations]

    # Run point-based inference
    masks = sam3.process_image_points(pil_image, points, labels)

    # Prepare points metadata for COCO export (x, y, is_positive)
    points_metadata = [(ann.point_x, ann.point_y, ann.is_positive) for ann in point_annotations]

    # Save mask (bboxes computed from mask, points stored in metadata)
    result.mask_blob_path = save_mask_to_storage(storage, masks, result.id)
    result.coco_json_blob_path = save_coco_to_storage(storage, image, [], masks, result.id, points=points_metadata)

    logger.info(f"Point mode processed {len(points)} points for image {image.id}")
    return True
