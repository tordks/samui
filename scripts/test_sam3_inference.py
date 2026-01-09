#!/usr/bin/env python3
"""Manual test script for SAM3 inference.

Usage examples:
    # Inside-box mode: segment within provided bboxes
    python scripts/test_sam3_inference.py --mode inside-box --image scripts/data/test.jpg --coco boxes.json

    # Find-all mode: text prompt only
    python scripts/test_sam3_inference.py --mode find-all --image scripts/data/test.jpg --text "person"

    # Find-all mode: exemplar boxes only
    python scripts/test_sam3_inference.py --mode find-all --image scripts/data/test.jpg --coco exemplars.json

    # Find-all mode: text + exemplar boxes
    python scripts/test_sam3_inference.py --mode find-all --image scripts/data/test.jpg --text "cat" --coco exemplars.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

# Add backend package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages/samui-backend/src"))

from samui_backend.services.sam3_inference import FindAllResult, SAM3Service

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_coco_bboxes(coco_path: Path) -> list[tuple[int, int, int, int]]:
    """Parse COCO JSON and extract bboxes as (x, y, w, h)."""
    with open(coco_path) as f:
        data = json.load(f)

    bboxes = []
    for ann in data.get("annotations", []):
        bbox = ann.get("bbox")
        if bbox and len(bbox) == 4:
            x, y, w, h = bbox
            bboxes.append((int(x), int(y), int(w), int(h)))

    return bboxes


def parse_coco_exemplars(coco_path: Path) -> list[tuple[tuple[int, int, int, int], bool]]:
    """Parse COCO JSON as exemplar boxes with polarity based on category_id.

    category_id > 0 or absent -> positive exemplar
    category_id <= 0 -> negative exemplar
    """
    with open(coco_path) as f:
        data = json.load(f)

    exemplars = []
    for ann in data.get("annotations", []):
        bbox = ann.get("bbox")
        if bbox and len(bbox) == 4:
            x, y, w, h = bbox
            category_id = ann.get("category_id", 1)
            is_positive = category_id > 0
            exemplars.append(((int(x), int(y), int(w), int(h)), is_positive))

    return exemplars


def create_overlay(image: Image.Image, masks: NDArray[np.uint8]) -> Image.Image:
    """Create visualization with colored mask overlays."""
    # Convert to RGBA for blending
    overlay = image.convert("RGBA")
    overlay_array = np.array(overlay)
    img_h, img_w = overlay_array.shape[:2]

    # Color palette for masks
    colors = [
        (255, 0, 0, 128),  # Red
        (0, 255, 0, 128),  # Green
        (0, 0, 255, 128),  # Blue
        (255, 255, 0, 128),  # Yellow
        (255, 0, 255, 128),  # Magenta
        (0, 255, 255, 128),  # Cyan
        (255, 128, 0, 128),  # Orange
        (128, 0, 255, 128),  # Purple
    ]

    for i, mask in enumerate(masks):
        color = colors[i % len(colors)]
        if mask.shape != (img_h, img_w):
            logger.warning(f"Mask {i} shape {mask.shape} doesn't match image {(img_h, img_w)}, skipping")
            continue
        # Create colored mask
        mask_bool = mask > 127
        for c in range(3):
            overlay_array[:, :, c] = np.where(
                mask_bool,
                (overlay_array[:, :, c] * 0.5 + color[c] * 0.5).astype(np.uint8),
                overlay_array[:, :, c],
            )

    return Image.fromarray(overlay_array).convert("RGB")


def save_results(
    masks: NDArray[np.uint8],
    output_dir: Path,
    image: Image.Image,
    scores: NDArray[np.float32] | None = None,
    bboxes: list[tuple[int, int, int, int]] | None = None,
) -> None:
    """Save mask PNGs and overlay visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save individual masks
    for i, mask in enumerate(masks):
        mask_path = output_dir / f"mask_{i}.png"
        Image.fromarray(mask).save(mask_path)
        logger.info(f"Saved mask to {mask_path}")

    # Save overlay visualization
    if len(masks) > 0:
        overlay = create_overlay(image, masks)
        overlay_path = output_dir / "overlay.png"
        overlay.save(overlay_path)
        logger.info(f"Saved overlay to {overlay_path}")

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"Results saved to: {output_dir}")
    print(f"Number of masks: {len(masks)}")

    if scores is not None and len(scores) > 0:
        print(f"Confidence scores: {[f'{s:.3f}' for s in scores]}")

    if bboxes:
        print(f"Bounding boxes (x, y, w, h): {bboxes}")

    print(f"{'=' * 50}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test SAM3 inference manually",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["inside-box", "find-all"],
        help="Segmentation mode: inside-box (segment within boxes) or find-all (discover objects)",
    )
    parser.add_argument("--image", required=True, type=Path, help="Path to input image")
    parser.add_argument("--text", type=str, help="Text prompt for find-all mode")
    parser.add_argument("--coco", type=Path, help="Path to COCO JSON with bbox annotations")
    parser.add_argument("--output-dir", type=Path, default=Path("./output"), help="Output directory for results")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold for find-all mode (0-1)")
    args = parser.parse_args()

    # Validate arguments
    if not args.image.exists():
        parser.error(f"Image file not found: {args.image}")

    if args.mode == "inside-box":
        if not args.coco:
            parser.error("inside-box mode requires --coco argument")
        if not args.coco.exists():
            parser.error(f"COCO file not found: {args.coco}")
    elif args.mode == "find-all":
        if not args.text and not args.coco:
            parser.error("find-all mode requires --text and/or --coco argument")
        if args.coco and not args.coco.exists():
            parser.error(f"COCO file not found: {args.coco}")

    # Load image
    logger.info(f"Loading image: {args.image}")
    image = Image.open(args.image).convert("RGB")
    logger.info(f"Image size: {image.size[0]}x{image.size[1]}")

    # Initialize and load model
    sam3 = SAM3Service()
    logger.info("Loading SAM3 model...")
    sam3.load_model()

    try:
        if args.mode == "inside-box":
            # Parse bboxes and run inside-box inference
            bboxes = parse_coco_bboxes(args.coco)
            logger.info(f"Parsed {len(bboxes)} bounding boxes from COCO JSON")

            if not bboxes:
                logger.warning("No bounding boxes found in COCO JSON")
                return

            logger.info("Running inside-box segmentation...")
            masks = sam3.process_image(image, bboxes)

            save_results(masks, args.output_dir, image, bboxes=bboxes)

        else:  # find-all mode
            # Parse exemplars if provided
            exemplars = None
            if args.coco:
                exemplars = parse_coco_exemplars(args.coco)
                logger.info(f"Parsed {len(exemplars)} exemplar boxes from COCO JSON")
                pos_count = sum(1 for _, is_pos in exemplars if is_pos)
                neg_count = len(exemplars) - pos_count
                logger.info(f"  Positive exemplars: {pos_count}, Negative exemplars: {neg_count}")

            logger.info(f"Running find-all segmentation (text={args.text!r}, threshold={args.threshold})...")
            result: FindAllResult = sam3.process_image_find_all(
                image,
                text_prompt=args.text,
                exemplar_boxes=exemplars,
                detection_threshold=args.threshold,
            )

            save_results(result.masks, args.output_dir, image, scores=result.scores, bboxes=result.bboxes)

    finally:
        logger.info("Unloading SAM3 model...")
        sam3.unload_model()


if __name__ == "__main__":
    main()
