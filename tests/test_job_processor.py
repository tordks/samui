"""Tests for job processor service."""

import uuid
from datetime import UTC, datetime

from samui_backend.db.models import BboxAnnotation, Image, PointAnnotation, ProcessingJob, ProcessingResult
from samui_backend.enums import JobStatus, PromptType, SegmentationMode
from samui_backend.services.job_processor import (
    build_annotations_snapshot,
    cleanup_stale_jobs,
    filter_images_needing_processing,
    get_point_annotations_for_image,
)
from sqlalchemy.orm import Session


def create_test_image(db: Session, filename: str = "test.jpg") -> Image:
    """Create a test image in the database."""
    image = Image(
        filename=filename,
        blob_path=f"images/{filename}",
        width=100,
        height=100,
    )
    db.add(image)
    db.commit()
    return image


def create_test_annotation(
    db: Session,
    image_id: uuid.UUID,
    prompt_type: PromptType = PromptType.SEGMENT,
) -> BboxAnnotation:
    """Create a test annotation in the database."""
    annotation = BboxAnnotation(
        image_id=image_id,
        bbox_x=10,
        bbox_y=10,
        bbox_width=50,
        bbox_height=50,
        prompt_type=prompt_type,
    )
    db.add(annotation)
    db.commit()
    return annotation


def create_test_job(
    db: Session,
    image_ids: list[uuid.UUID],
    mode: SegmentationMode = SegmentationMode.INSIDE_BOX,
    status: JobStatus = JobStatus.QUEUED,
    filenames: list[str] | None = None,
    annotations_snapshot: dict | None = None,
) -> ProcessingJob:
    """Create a test processing job."""
    job = ProcessingJob(
        mode=mode,
        status=status,
        image_ids=[str(img_id) for img_id in image_ids],
        image_filenames=filenames or [f"test_{i}.png" for i in range(len(image_ids))],
        annotations_snapshot=annotations_snapshot,
    )
    db.add(job)
    db.commit()
    return job


def create_test_result(
    db: Session,
    job_id: uuid.UUID,
    image_id: uuid.UUID,
    mode: SegmentationMode = SegmentationMode.INSIDE_BOX,
) -> ProcessingResult:
    """Create a test processing result."""
    result = ProcessingResult(
        job_id=job_id,
        image_id=image_id,
        mode=mode,
        mask_blob_path=f"masks/{uuid.uuid4()}.png",
        coco_json_blob_path=f"coco/{uuid.uuid4()}.json",
    )
    db.add(result)
    db.commit()
    return result


def make_bbox_snapshot(ann: BboxAnnotation) -> dict:
    """Create a snapshot dict from a BboxAnnotation."""
    return {
        "id": str(ann.id),
        "bbox_x": ann.bbox_x,
        "bbox_y": ann.bbox_y,
        "bbox_width": ann.bbox_width,
        "bbox_height": ann.bbox_height,
        "prompt_type": ann.prompt_type.value,
    }


def make_point_snapshot(ann: PointAnnotation) -> dict:
    """Create a snapshot dict from a PointAnnotation."""
    return {
        "id": str(ann.id),
        "point_x": ann.point_x,
        "point_y": ann.point_y,
        "is_positive": ann.is_positive,
    }


class TestFilterImagesNeedingProcessing:
    """Tests for filter_images_needing_processing function."""

    def test_no_annotations_no_job_returns_empty(self, db_session: Session) -> None:
        """Image with no annotations and no previous job is not included."""
        image = create_test_image(db_session)
        snapshot = build_annotations_snapshot(db_session, image, SegmentationMode.INSIDE_BOX)

        result = filter_images_needing_processing(db_session, {image.id: snapshot}, SegmentationMode.INSIDE_BOX)
        assert image.id not in result

    def test_has_annotations_no_job_returns_image(self, db_session: Session) -> None:
        """Image with annotations but no previous job is included."""
        image = create_test_image(db_session)
        create_test_annotation(db_session, image.id)
        snapshot = build_annotations_snapshot(db_session, image, SegmentationMode.INSIDE_BOX)

        result = filter_images_needing_processing(db_session, {image.id: snapshot}, SegmentationMode.INSIDE_BOX)
        assert image.id in result

    def test_find_all_text_prompt_only_returns_image(self, db_session: Session) -> None:
        """Find-all mode with text prompt but no exemplars includes image."""
        image = create_test_image(db_session)
        image.text_prompt = "find all cats"
        db_session.commit()
        snapshot = build_annotations_snapshot(db_session, image, SegmentationMode.FIND_ALL)

        result = filter_images_needing_processing(db_session, {image.id: snapshot}, SegmentationMode.FIND_ALL)
        assert image.id in result

    def test_annotations_unchanged_returns_empty(self, db_session: Session) -> None:
        """Same annotation set as previous job excludes image."""
        image = create_test_image(db_session)
        ann = create_test_annotation(db_session, image.id)
        job_snapshot = {
            str(image.id): {
                "text_prompt": None,
                "bbox_annotations": [make_bbox_snapshot(ann)],
                "point_annotations": [],
            }
        }
        create_test_job(db_session, [image.id], annotations_snapshot=job_snapshot)
        snapshot = build_annotations_snapshot(db_session, image, SegmentationMode.INSIDE_BOX)

        result = filter_images_needing_processing(db_session, {image.id: snapshot}, SegmentationMode.INSIDE_BOX)
        assert image.id not in result

    def test_annotation_added_returns_image(self, db_session: Session) -> None:
        """New annotation added since last job includes image."""
        image = create_test_image(db_session)
        ann1 = create_test_annotation(db_session, image.id)
        job_snapshot = {
            str(image.id): {
                "text_prompt": None,
                "bbox_annotations": [make_bbox_snapshot(ann1)],
                "point_annotations": [],
            }
        }
        create_test_job(db_session, [image.id], annotations_snapshot=job_snapshot)

        # Add another annotation
        create_test_annotation(db_session, image.id)
        snapshot = build_annotations_snapshot(db_session, image, SegmentationMode.INSIDE_BOX)

        result = filter_images_needing_processing(db_session, {image.id: snapshot}, SegmentationMode.INSIDE_BOX)
        assert image.id in result

    def test_annotation_deleted_returns_image(self, db_session: Session) -> None:
        """Annotation deleted since last job includes image."""
        image = create_test_image(db_session)
        ann1 = create_test_annotation(db_session, image.id)
        ann2 = create_test_annotation(db_session, image.id)
        job_snapshot = {
            str(image.id): {
                "text_prompt": None,
                "bbox_annotations": [make_bbox_snapshot(ann1), make_bbox_snapshot(ann2)],
                "point_annotations": [],
            }
        }
        create_test_job(db_session, [image.id], annotations_snapshot=job_snapshot)

        # Delete one annotation
        db_session.delete(ann2)
        db_session.commit()
        snapshot = build_annotations_snapshot(db_session, image, SegmentationMode.INSIDE_BOX)

        result = filter_images_needing_processing(db_session, {image.id: snapshot}, SegmentationMode.INSIDE_BOX)
        assert image.id in result

    def test_text_prompt_changed_returns_image(self, db_session: Session) -> None:
        """Changed text prompt in find-all mode includes image."""
        image = create_test_image(db_session)
        image.text_prompt = "find all dogs"
        db_session.commit()

        job_snapshot = {
            str(image.id): {
                "text_prompt": "find all cats",  # Different from current
                "bbox_annotations": [],
                "point_annotations": [],
            }
        }
        create_test_job(db_session, [image.id], mode=SegmentationMode.FIND_ALL, annotations_snapshot=job_snapshot)
        snapshot = build_annotations_snapshot(db_session, image, SegmentationMode.FIND_ALL)

        result = filter_images_needing_processing(db_session, {image.id: snapshot}, SegmentationMode.FIND_ALL)
        assert image.id in result

    def test_text_prompt_unchanged_returns_empty(self, db_session: Session) -> None:
        """Same text prompt in find-all mode excludes image."""
        image = create_test_image(db_session)
        image.text_prompt = "find all cats"
        db_session.commit()

        job_snapshot = {
            str(image.id): {
                "text_prompt": "find all cats",  # Same as current
                "bbox_annotations": [],
                "point_annotations": [],
            }
        }
        create_test_job(db_session, [image.id], mode=SegmentationMode.FIND_ALL, annotations_snapshot=job_snapshot)
        snapshot = build_annotations_snapshot(db_session, image, SegmentationMode.FIND_ALL)

        result = filter_images_needing_processing(db_session, {image.id: snapshot}, SegmentationMode.FIND_ALL)
        assert image.id not in result

    def test_inside_box_ignores_exemplar_annotations(self, db_session: Session) -> None:
        """Inside-box mode ignores exemplar annotations."""
        image = create_test_image(db_session)
        # Add exemplar annotation (should be ignored in inside-box mode)
        create_test_annotation(db_session, image.id, prompt_type=PromptType.POSITIVE_EXEMPLAR)
        snapshot = build_annotations_snapshot(db_session, image, SegmentationMode.INSIDE_BOX)

        result = filter_images_needing_processing(db_session, {image.id: snapshot}, SegmentationMode.INSIDE_BOX)
        assert image.id not in result  # No SEGMENT annotations

    def test_find_all_ignores_segment_annotations(self, db_session: Session) -> None:
        """Find-all mode ignores segment annotations."""
        image = create_test_image(db_session)
        # Add segment annotation (should be ignored in find-all mode)
        create_test_annotation(db_session, image.id, prompt_type=PromptType.SEGMENT)
        snapshot = build_annotations_snapshot(db_session, image, SegmentationMode.FIND_ALL)

        result = filter_images_needing_processing(db_session, {image.id: snapshot}, SegmentationMode.FIND_ALL)
        assert image.id not in result  # No exemplars and no text prompt

    def test_pending_job_with_same_annotations_returns_empty(self, db_session: Session) -> None:
        """Image not included when a pending job already has the same annotations."""
        image = create_test_image(db_session)
        ann = create_test_annotation(db_session, image.id)
        job_snapshot = {
            str(image.id): {
                "text_prompt": None,
                "bbox_annotations": [make_bbox_snapshot(ann)],
                "point_annotations": [],
            }
        }
        # Job is QUEUED (pending), not completed
        create_test_job(db_session, [image.id], status=JobStatus.QUEUED, annotations_snapshot=job_snapshot)
        snapshot = build_annotations_snapshot(db_session, image, SegmentationMode.INSIDE_BOX)

        result = filter_images_needing_processing(db_session, {image.id: snapshot}, SegmentationMode.INSIDE_BOX)
        assert image.id not in result  # Already queued with same annotations

    def test_filters_multiple_images(self, db_session: Session) -> None:
        """Correctly filters multiple images in a single call."""
        # Image 1: needs processing (has annotation, no job)
        img1 = create_test_image(db_session, "test1.jpg")
        create_test_annotation(db_session, img1.id)

        # Image 2: doesn't need processing (has job with same annotations)
        img2 = create_test_image(db_session, "test2.jpg")
        ann2 = create_test_annotation(db_session, img2.id)
        job_snapshot = {
            str(img2.id): {
                "text_prompt": None,
                "bbox_annotations": [make_bbox_snapshot(ann2)],
                "point_annotations": [],
            }
        }
        create_test_job(db_session, [img2.id], annotations_snapshot=job_snapshot)

        # Image 3: doesn't need processing (no annotations)
        img3 = create_test_image(db_session, "test3.jpg")

        snapshots = {
            img1.id: build_annotations_snapshot(db_session, img1, SegmentationMode.INSIDE_BOX),
            img2.id: build_annotations_snapshot(db_session, img2, SegmentationMode.INSIDE_BOX),
            img3.id: build_annotations_snapshot(db_session, img3, SegmentationMode.INSIDE_BOX),
        }

        result = filter_images_needing_processing(db_session, snapshots, SegmentationMode.INSIDE_BOX)
        assert result == [img1.id]


class TestCleanupStaleJobs:
    """Tests for cleanup_stale_jobs function."""

    def test_resets_running_jobs_to_failed(self, db_session: Session) -> None:
        """Running jobs are reset to failed."""
        image = create_test_image(db_session)
        job = create_test_job(db_session, [image.id], status=JobStatus.RUNNING)
        job.started_at = datetime.now(UTC)
        db_session.commit()

        count = cleanup_stale_jobs(db_session)

        db_session.refresh(job)
        assert count == 1
        assert job.status == JobStatus.FAILED
        assert job.error == "Server restarted while job was running"
        assert job.completed_at is not None

    def test_leaves_queued_jobs_unchanged(self, db_session: Session) -> None:
        """Queued jobs are not affected."""
        image = create_test_image(db_session)
        job = create_test_job(db_session, [image.id], status=JobStatus.QUEUED)

        count = cleanup_stale_jobs(db_session)

        db_session.refresh(job)
        assert count == 0
        assert job.status == JobStatus.QUEUED

    def test_leaves_completed_jobs_unchanged(self, db_session: Session) -> None:
        """Completed jobs are not affected."""
        image = create_test_image(db_session)
        job = create_test_job(db_session, [image.id], status=JobStatus.COMPLETED)
        job.completed_at = datetime.now(UTC)
        db_session.commit()

        count = cleanup_stale_jobs(db_session)

        db_session.refresh(job)
        assert count == 0
        assert job.status == JobStatus.COMPLETED

    def test_multiple_running_jobs(self, db_session: Session) -> None:
        """Multiple running jobs are all reset."""
        img1 = create_test_image(db_session, "test1.jpg")
        img2 = create_test_image(db_session, "test2.jpg")

        job1 = create_test_job(db_session, [img1.id], status=JobStatus.RUNNING)
        job2 = create_test_job(db_session, [img2.id], status=JobStatus.RUNNING)

        count = cleanup_stale_jobs(db_session)

        db_session.refresh(job1)
        db_session.refresh(job2)
        assert count == 2
        assert job1.status == JobStatus.FAILED
        assert job2.status == JobStatus.FAILED


def create_test_point_annotation(
    db: Session,
    image_id: uuid.UUID,
    point_x: int = 50,
    point_y: int = 50,
    is_positive: bool = True,
) -> PointAnnotation:
    """Create a test point annotation in the database."""
    annotation = PointAnnotation(
        image_id=image_id,
        point_x=point_x,
        point_y=point_y,
        is_positive=is_positive,
    )
    db.add(annotation)
    db.commit()
    return annotation


class TestGetPointAnnotationsForImage:
    """Tests for get_point_annotations_for_image function."""

    def test_returns_point_annotations_for_image(self, db_session: Session) -> None:
        """Returns all point annotations for the given image."""
        image = create_test_image(db_session)
        ann1 = create_test_point_annotation(db_session, image.id, point_x=25, point_y=25, is_positive=True)
        ann2 = create_test_point_annotation(db_session, image.id, point_x=75, point_y=75, is_positive=False)

        result = get_point_annotations_for_image(db_session, image.id)

        assert len(result) == 2
        assert {a.id for a in result} == {ann1.id, ann2.id}

    def test_returns_empty_when_no_points(self, db_session: Session) -> None:
        """Returns empty list when image has no point annotations."""
        image = create_test_image(db_session)

        result = get_point_annotations_for_image(db_session, image.id)

        assert result == []

    def test_only_returns_points_for_specified_image(self, db_session: Session) -> None:
        """Only returns point annotations for the specified image, not others."""
        img1 = create_test_image(db_session, "test1.jpg")
        img2 = create_test_image(db_session, "test2.jpg")

        ann1 = create_test_point_annotation(db_session, img1.id)
        create_test_point_annotation(db_session, img2.id)  # For different image

        result = get_point_annotations_for_image(db_session, img1.id)

        assert len(result) == 1
        assert result[0].id == ann1.id


class TestFilterImagesNeedingProcessingPointMode:
    """Tests for filter_images_needing_processing function with POINT mode."""

    def test_point_mode_no_points_no_job_returns_empty(self, db_session: Session) -> None:
        """POINT mode with no point annotations and no previous job excludes image."""
        image = create_test_image(db_session)
        snapshot = build_annotations_snapshot(db_session, image, SegmentationMode.POINT)

        result = filter_images_needing_processing(db_session, {image.id: snapshot}, SegmentationMode.POINT)
        assert image.id not in result

    def test_point_mode_has_points_no_job_returns_image(self, db_session: Session) -> None:
        """POINT mode with point annotations but no previous job includes image."""
        image = create_test_image(db_session)
        create_test_point_annotation(db_session, image.id)
        snapshot = build_annotations_snapshot(db_session, image, SegmentationMode.POINT)

        result = filter_images_needing_processing(db_session, {image.id: snapshot}, SegmentationMode.POINT)
        assert image.id in result

    def test_point_mode_points_unchanged_returns_empty(self, db_session: Session) -> None:
        """POINT mode with same point annotations as previous job excludes image."""
        image = create_test_image(db_session)
        ann = create_test_point_annotation(db_session, image.id)
        job_snapshot = {
            str(image.id): {
                "text_prompt": None,
                "bbox_annotations": [],
                "point_annotations": [make_point_snapshot(ann)],
            }
        }
        create_test_job(db_session, [image.id], mode=SegmentationMode.POINT, annotations_snapshot=job_snapshot)
        snapshot = build_annotations_snapshot(db_session, image, SegmentationMode.POINT)

        result = filter_images_needing_processing(db_session, {image.id: snapshot}, SegmentationMode.POINT)
        assert image.id not in result

    def test_point_mode_point_added_returns_image(self, db_session: Session) -> None:
        """POINT mode with new point annotation added since last job includes image."""
        image = create_test_image(db_session)
        ann1 = create_test_point_annotation(db_session, image.id, point_x=25, point_y=25)
        job_snapshot = {
            str(image.id): {
                "text_prompt": None,
                "bbox_annotations": [],
                "point_annotations": [make_point_snapshot(ann1)],
            }
        }
        create_test_job(db_session, [image.id], mode=SegmentationMode.POINT, annotations_snapshot=job_snapshot)

        # Add another point annotation
        create_test_point_annotation(db_session, image.id, point_x=75, point_y=75)
        snapshot = build_annotations_snapshot(db_session, image, SegmentationMode.POINT)

        result = filter_images_needing_processing(db_session, {image.id: snapshot}, SegmentationMode.POINT)
        assert image.id in result

    def test_point_mode_point_deleted_returns_image(self, db_session: Session) -> None:
        """POINT mode with point annotation deleted since last job includes image."""
        image = create_test_image(db_session)
        ann1 = create_test_point_annotation(db_session, image.id, point_x=25, point_y=25)
        ann2 = create_test_point_annotation(db_session, image.id, point_x=75, point_y=75)
        job_snapshot = {
            str(image.id): {
                "text_prompt": None,
                "bbox_annotations": [],
                "point_annotations": [make_point_snapshot(ann1), make_point_snapshot(ann2)],
            }
        }
        create_test_job(db_session, [image.id], mode=SegmentationMode.POINT, annotations_snapshot=job_snapshot)

        # Delete one point annotation
        db_session.delete(ann2)
        db_session.commit()
        snapshot = build_annotations_snapshot(db_session, image, SegmentationMode.POINT)

        result = filter_images_needing_processing(db_session, {image.id: snapshot}, SegmentationMode.POINT)
        assert image.id in result

    def test_point_mode_ignores_bbox_annotations(self, db_session: Session) -> None:
        """POINT mode ignores bbox annotations."""
        image = create_test_image(db_session)
        # Add bbox annotation (should be ignored in point mode)
        create_test_annotation(db_session, image.id, prompt_type=PromptType.SEGMENT)
        snapshot = build_annotations_snapshot(db_session, image, SegmentationMode.POINT)

        result = filter_images_needing_processing(db_session, {image.id: snapshot}, SegmentationMode.POINT)
        assert image.id not in result  # No point annotations

    def test_inside_box_mode_ignores_point_annotations(self, db_session: Session) -> None:
        """Inside-box mode ignores point annotations."""
        image = create_test_image(db_session)
        # Add point annotation (should be ignored in inside-box mode)
        create_test_point_annotation(db_session, image.id)
        snapshot = build_annotations_snapshot(db_session, image, SegmentationMode.INSIDE_BOX)

        result = filter_images_needing_processing(db_session, {image.id: snapshot}, SegmentationMode.INSIDE_BOX)
        assert image.id not in result  # No bbox annotations
