"""Snapshot manager for annotation state comparison and filtering."""

from __future__ import annotations

import uuid

from sqlalchemy import desc
from sqlalchemy.orm import Session

from samui_backend.db.models import BboxAnnotation, Image, PointAnnotation, ProcessingJob
from samui_backend.enums import PromptType, SegmentationMode
from samui_backend.schemas import AnnotationsSnapshot, BboxAnnotationSnapshot, PointAnnotationSnapshot


def get_annotations_for_mode(db: Session, image_id: uuid.UUID, mode: SegmentationMode) -> list[BboxAnnotation]:
    """Get bbox annotations relevant to the given segmentation mode."""
    if mode == SegmentationMode.INSIDE_BOX:
        return (
            db.query(BboxAnnotation)
            .filter(BboxAnnotation.image_id == image_id, BboxAnnotation.prompt_type == PromptType.SEGMENT)
            .all()
        )
    elif mode == SegmentationMode.FIND_ALL:
        # Find-all mode uses exemplar annotations
        return (
            db.query(BboxAnnotation)
            .filter(
                BboxAnnotation.image_id == image_id,
                BboxAnnotation.prompt_type.in_([PromptType.POSITIVE_EXEMPLAR, PromptType.NEGATIVE_EXEMPLAR]),
            )
            .all()
        )
    else:
        # POINT mode doesn't use BboxAnnotation
        return []


def get_point_annotations_for_image(db: Session, image_id: uuid.UUID) -> list[PointAnnotation]:
    """Get point annotations for an image (used in POINT mode)."""
    return db.query(PointAnnotation).filter(PointAnnotation.image_id == image_id).all()


def build_annotations_snapshot(db: Session, image: Image, mode: SegmentationMode) -> AnnotationsSnapshot:
    """Build an annotations snapshot for an image based on the segmentation mode."""
    bbox_snapshots: list[BboxAnnotationSnapshot] = []
    point_snapshots: list[PointAnnotationSnapshot] = []

    if mode == SegmentationMode.POINT:
        point_annotations = get_point_annotations_for_image(db, image.id)
        point_snapshots = [
            PointAnnotationSnapshot(
                id=ann.id,
                point_x=ann.point_x,
                point_y=ann.point_y,
                is_positive=ann.is_positive,
            )
            for ann in point_annotations
        ]
    else:
        bbox_annotations = get_annotations_for_mode(db, image.id, mode)
        bbox_snapshots = [
            BboxAnnotationSnapshot(
                id=ann.id,
                bbox_x=ann.bbox_x,
                bbox_y=ann.bbox_y,
                bbox_width=ann.bbox_width,
                bbox_height=ann.bbox_height,
                prompt_type=ann.prompt_type,
            )
            for ann in bbox_annotations
        ]

    return AnnotationsSnapshot(
        text_prompt=image.text_prompt,
        bbox_annotations=bbox_snapshots,
        point_annotations=point_snapshots,
    )


def _get_snapshot_annotation_ids(snapshot: AnnotationsSnapshot, mode: SegmentationMode) -> set[str]:
    """Extract annotation IDs from a snapshot based on mode."""
    if mode == SegmentationMode.POINT:
        return {str(a.id) for a in snapshot.point_annotations}
    else:
        return {str(a.id) for a in snapshot.bbox_annotations}


def _check_image_needs_processing(
    image_id_str: str,
    snapshot: AnnotationsSnapshot,
    mode: SegmentationMode,
    jobs: list[ProcessingJob],
) -> bool:
    """Check if a single image needs processing given pre-fetched jobs."""
    current_ids = _get_snapshot_annotation_ids(snapshot, mode)

    # Find most recent job that includes this image
    latest_job = next((job for job in jobs if image_id_str in job.image_ids), None)

    # No previous job - process if there's something to process
    if not latest_job:
        has_annotations = bool(current_ids)
        has_find_all_prompt = mode == SegmentationMode.FIND_ALL and bool(snapshot.text_prompt)
        return has_annotations or has_find_all_prompt

    # Compare current annotations against job's snapshot
    job_snapshot = latest_job.annotations_snapshot or {}
    last_snapshot_data = job_snapshot.get(image_id_str, {})

    if mode == SegmentationMode.POINT:
        last_ids = {str(a["id"]) for a in last_snapshot_data.get("point_annotations", [])}
    else:
        last_ids = {str(a["id"]) for a in last_snapshot_data.get("bbox_annotations", [])}

    if current_ids != last_ids:
        return True

    # Check text_prompt changed (find-all mode)
    last_text_prompt = last_snapshot_data.get("text_prompt")
    return mode == SegmentationMode.FIND_ALL and snapshot.text_prompt != last_text_prompt


def filter_images_needing_processing(
    db: Session,
    snapshots: dict[uuid.UUID, AnnotationsSnapshot],
    mode: SegmentationMode,
) -> list[uuid.UUID]:
    """Filter images to those needing processing (batch version).

    Fetches jobs once and checks all images efficiently.

    Args:
        db: Database session.
        snapshots: Dict mapping image_id to its annotation snapshot.
        mode: Segmentation mode.

    Returns:
        List of image IDs that need processing.
    """
    if not snapshots:
        return []

    # Fetch all jobs for this mode once
    jobs = db.query(ProcessingJob).filter(ProcessingJob.mode == mode).order_by(desc(ProcessingJob.created_at)).all()

    return [
        image_id
        for image_id, snapshot in snapshots.items()
        if _check_image_needs_processing(str(image_id), snapshot, mode, jobs)
    ]
