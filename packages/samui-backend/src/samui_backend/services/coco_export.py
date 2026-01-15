"""COCO JSON export service for generating annotation files."""

import uuid
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _mask_to_rle(mask: NDArray[np.uint8]) -> dict[str, Any]:
    """Convert a binary mask to COCO RLE format.

    Args:
        mask: Binary mask array of shape (height, width) with values 0 or 255.

    Returns:
        COCO RLE dict with 'counts' (list of run lengths) and 'size' [height, width].
    """
    # Flatten in column-major (Fortran) order as COCO expects
    pixels = (mask.flatten(order="F") > 0).astype(np.uint8)

    # Compute run-length encoding
    # Start with a run of zeros if first pixel is 1
    runs = []
    prev = 0
    count = 0

    for pixel in pixels:
        if pixel == prev:
            count += 1
        else:
            runs.append(count)
            count = 1
            prev = pixel

    # Append final run
    runs.append(count)

    # If mask starts with 1, prepend a 0-length run of 0s
    if len(runs) > 0 and pixels[0] == 1:
        runs.insert(0, 0)

    return {"counts": runs, "size": [mask.shape[0], mask.shape[1]]}


def _compute_bbox_area(mask: NDArray[np.uint8]) -> int:
    """Compute the area of a binary mask.

    Args:
        mask: Binary mask array.

    Returns:
        Number of pixels in the mask.
    """
    return int(np.sum(mask > 0))


def _compute_bbox_from_mask(mask: NDArray[np.uint8]) -> tuple[int, int, int, int]:
    """Compute bounding box from a binary mask.

    Args:
        mask: Binary mask array of shape (height, width).

    Returns:
        Bounding box as (x, y, width, height).
    """
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)

    if not np.any(rows) or not np.any(cols):
        return (0, 0, 0, 0)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return (int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1))


def generate_coco_json(  # noqa: PLR0913
    image_id: uuid.UUID,
    filename: str,
    width: int,
    height: int,
    bboxes: list[tuple[int, int, int, int]],
    masks: NDArray[np.uint8],
    category_name: str = "object",
    points: list[tuple[int, int, bool]] | None = None,
) -> dict[str, Any]:
    """Generate COCO JSON for a single image with its annotations.

    Args:
        image_id: Unique identifier for the image.
        filename: Original image filename.
        width: Image width in pixels.
        height: Image height in pixels.
        bboxes: List of bounding boxes in (x, y, width, height) format.
            If empty but masks provided, bboxes are computed from masks.
        masks: Array of binary masks with shape (num_annotations, height, width).
        category_name: Name for the segmentation category.
        points: Optional list of (x, y, is_positive) tuples for point annotations.
            Stored in annotation metadata when provided.

    Returns:
        COCO-format dict with images, annotations, and categories arrays.
    """
    # Create COCO structure
    coco: dict[str, Any] = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": category_name, "supercategory": "none"}],
    }

    # Add image info
    coco["images"].append({
        "id": str(image_id),
        "file_name": filename,
        "width": width,
        "height": height,
    })

    # If bboxes not provided, compute from masks
    if not bboxes and masks.size > 0:
        bboxes = [_compute_bbox_from_mask(mask) for mask in masks]

    # Add annotations
    for idx, (bbox, mask) in enumerate(zip(bboxes, masks, strict=True)):
        x, y, w, h = bbox

        # Compute segmentation RLE and area from mask
        segmentation = _mask_to_rle(mask)
        area = _compute_bbox_area(mask)

        annotation: dict[str, Any] = {
            "id": idx + 1,
            "image_id": str(image_id),
            "category_id": 1,
            "segmentation": segmentation,
            "bbox": [x, y, w, h],  # COCO uses [x, y, width, height]
            "area": area,
            "iscrowd": 0,
        }

        # Add point prompts metadata if provided
        if points:
            annotation["point_prompts"] = [{"x": px, "y": py, "is_positive": is_pos} for px, py, is_pos in points]

        coco["annotations"].append(annotation)

    return coco
