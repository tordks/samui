"""Tests for job processor service."""

import uuid
from datetime import UTC, datetime

from samui_backend.db.models import BboxAnnotation, Image, ProcessingJob, ProcessingResult
from samui_backend.enums import JobStatus, PromptType, SegmentationMode
from samui_backend.services.job_processor import (
    cleanup_stale_jobs,
    get_images_needing_processing,
    needs_processing,
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
) -> ProcessingJob:
    """Create a test processing job."""
    job = ProcessingJob(
        mode=mode,
        status=status,
        image_ids=[str(img_id) for img_id in image_ids],
        image_filenames=filenames or [f"test_{i}.png" for i in range(len(image_ids))],
    )
    db.add(job)
    db.commit()
    return job


def create_test_result(
    db: Session,
    job_id: uuid.UUID,
    image_id: uuid.UUID,
    mode: SegmentationMode = SegmentationMode.INSIDE_BOX,
    annotation_ids: list[str] | None = None,
    text_prompt_used: str | None = None,
) -> ProcessingResult:
    """Create a test processing result."""
    result = ProcessingResult(
        job_id=job_id,
        image_id=image_id,
        mode=mode,
        mask_blob_path=f"masks/{uuid.uuid4()}.png",
        coco_json_blob_path=f"coco/{uuid.uuid4()}.json",
        annotation_ids=annotation_ids,
        text_prompt_used=text_prompt_used,
    )
    db.add(result)
    db.commit()
    return result


class TestNeedsProcessing:
    """Tests for needs_processing function."""

    def test_no_image_returns_false(self, db_session: Session) -> None:
        """Non-existent image returns False."""
        result = needs_processing(db_session, uuid.uuid4(), SegmentationMode.INSIDE_BOX)
        assert result is False

    def test_no_annotations_no_result_returns_false(self, db_session: Session) -> None:
        """Image with no annotations and no previous result returns False."""
        image = create_test_image(db_session)
        result = needs_processing(db_session, image.id, SegmentationMode.INSIDE_BOX)
        assert result is False

    def test_has_annotations_no_result_returns_true(self, db_session: Session) -> None:
        """Image with annotations but no previous result returns True."""
        image = create_test_image(db_session)
        create_test_annotation(db_session, image.id)

        result = needs_processing(db_session, image.id, SegmentationMode.INSIDE_BOX)
        assert result is True

    def test_find_all_text_prompt_only_returns_true(self, db_session: Session) -> None:
        """Find-all mode with text prompt but no exemplars returns True."""
        image = create_test_image(db_session)
        image.text_prompt = "find all cats"
        db_session.commit()

        result = needs_processing(db_session, image.id, SegmentationMode.FIND_ALL)
        assert result is True

    def test_annotations_unchanged_returns_false(self, db_session: Session) -> None:
        """Same annotation set as previous result returns False."""
        image = create_test_image(db_session)
        ann = create_test_annotation(db_session, image.id)
        job = create_test_job(db_session, [image.id])
        create_test_result(db_session, job.id, image.id, annotation_ids=[str(ann.id)])

        result = needs_processing(db_session, image.id, SegmentationMode.INSIDE_BOX)
        assert result is False

    def test_annotation_added_returns_true(self, db_session: Session) -> None:
        """New annotation added since last result returns True."""
        image = create_test_image(db_session)
        ann1 = create_test_annotation(db_session, image.id)
        job = create_test_job(db_session, [image.id])
        create_test_result(db_session, job.id, image.id, annotation_ids=[str(ann1.id)])

        # Add another annotation
        create_test_annotation(db_session, image.id)

        result = needs_processing(db_session, image.id, SegmentationMode.INSIDE_BOX)
        assert result is True

    def test_annotation_deleted_returns_true(self, db_session: Session) -> None:
        """Annotation deleted since last result returns True."""
        image = create_test_image(db_session)
        ann1 = create_test_annotation(db_session, image.id)
        ann2 = create_test_annotation(db_session, image.id)
        job = create_test_job(db_session, [image.id])
        create_test_result(db_session, job.id, image.id, annotation_ids=[str(ann1.id), str(ann2.id)])

        # Delete one annotation
        db_session.delete(ann2)
        db_session.commit()

        result = needs_processing(db_session, image.id, SegmentationMode.INSIDE_BOX)
        assert result is True

    def test_text_prompt_changed_returns_true(self, db_session: Session) -> None:
        """Changed text prompt in find-all mode returns True."""
        image = create_test_image(db_session)
        image.text_prompt = "find all dogs"
        db_session.commit()

        job = create_test_job(db_session, [image.id], mode=SegmentationMode.FIND_ALL)
        create_test_result(
            db_session,
            job.id,
            image.id,
            mode=SegmentationMode.FIND_ALL,
            text_prompt_used="find all cats",
        )

        result = needs_processing(db_session, image.id, SegmentationMode.FIND_ALL)
        assert result is True

    def test_text_prompt_unchanged_returns_false(self, db_session: Session) -> None:
        """Same text prompt in find-all mode returns False."""
        image = create_test_image(db_session)
        image.text_prompt = "find all cats"
        db_session.commit()

        job = create_test_job(db_session, [image.id], mode=SegmentationMode.FIND_ALL)
        create_test_result(
            db_session,
            job.id,
            image.id,
            mode=SegmentationMode.FIND_ALL,
            text_prompt_used="find all cats",
        )

        result = needs_processing(db_session, image.id, SegmentationMode.FIND_ALL)
        assert result is False

    def test_inside_box_ignores_exemplar_annotations(self, db_session: Session) -> None:
        """Inside-box mode ignores exemplar annotations."""
        image = create_test_image(db_session)
        # Add exemplar annotation (should be ignored in inside-box mode)
        create_test_annotation(db_session, image.id, prompt_type=PromptType.POSITIVE_EXEMPLAR)

        result = needs_processing(db_session, image.id, SegmentationMode.INSIDE_BOX)
        assert result is False  # No SEGMENT annotations

    def test_find_all_ignores_segment_annotations(self, db_session: Session) -> None:
        """Find-all mode ignores segment annotations."""
        image = create_test_image(db_session)
        # Add segment annotation (should be ignored in find-all mode)
        create_test_annotation(db_session, image.id, prompt_type=PromptType.SEGMENT)

        result = needs_processing(db_session, image.id, SegmentationMode.FIND_ALL)
        assert result is False  # No exemplars and no text prompt


class TestGetImagesNeedingProcessing:
    """Tests for get_images_needing_processing function."""

    def test_filters_images_needing_processing(self, db_session: Session) -> None:
        """Returns only images that need processing."""
        # Image 1: needs processing (has annotation, no result)
        img1 = create_test_image(db_session, "test1.jpg")
        create_test_annotation(db_session, img1.id)

        # Image 2: doesn't need processing (has result with same annotations)
        img2 = create_test_image(db_session, "test2.jpg")
        ann2 = create_test_annotation(db_session, img2.id)
        job = create_test_job(db_session, [img2.id])
        create_test_result(db_session, job.id, img2.id, annotation_ids=[str(ann2.id)])

        # Image 3: doesn't need processing (no annotations)
        img3 = create_test_image(db_session, "test3.jpg")

        result = get_images_needing_processing(
            db_session,
            [img1.id, img2.id, img3.id],
            SegmentationMode.INSIDE_BOX,
        )

        assert result == [img1.id]

    def test_empty_list_returns_empty(self, db_session: Session) -> None:
        """Empty input returns empty list."""
        result = get_images_needing_processing(db_session, [], SegmentationMode.INSIDE_BOX)
        assert result == []


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
