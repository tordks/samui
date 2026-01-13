# Job-Based Processing Feature Tasklist

## Phase 1: Data Model Updates

**Goal:** Add ProcessingJob table and update related models.

**Deliverable:** New job model, updated result model, removed deprecated fields.

**Tasks:**

- [x] [P1.1] Add `JobStatus` enum to `enums.py` with values: queued, running, completed, failed
- [x] [P1.2] Remove `AnnotationSource` enum from `enums.py` (no longer needed)
- [x] [P1.3] Create `ProcessingJob` model in `db/models.py`
  - Fields: id, mode, status, image_ids (JSON), current_index, created_at, started_at, completed_at, error
  - Add relationship to ProcessingResult
- [x] [P1.4] Modify `ProcessingResult` in `db/models.py`: Add `job_id` foreign key
- [x] [P1.5] Modify `ProcessingResult` in `db/models.py`: Add `text_prompt_used` field (nullable string)
- [x] [P1.6] Modify `ProcessingResult` in `db/models.py`: Add `annotation_ids` field (JSON, list of UUID strings for input annotations)
- [x] [P1.7] Modify `ProcessingResult` in `db/models.py`: Add `bboxes` field (JSON, list of bbox dicts for SAM3 discoveries)
- [x] [P1.8] Modify `ProcessingResult` in `db/models.py`: Remove unique constraint on (image_id, mode)
- [x] [P1.9] Update blob path format: use `result_id` instead of `image_id` for history support
  - `masks/{result_id}.png` and `coco/{result_id}.json`
  - Update processing logic to generate paths using result_id
- [x] [P1.10] Modify `Image` in `db/models.py`: Remove `processing_status` field
- [x] [P1.11] Modify `Image` in `db/models.py`: Change `processing_result` relationship to `processing_results` (list)
- [x] [P1.12] Modify `Annotation` in `db/models.py`: Remove `source` field
- [x] [P1.13] Run test collection to verify models compile: `cd packages/samui-backend && uv run pytest ../../tests/ -v --collect-only`

**Checkpoints:**

- [x] Code quality: Run `uvx ruff check packages/samui-backend/`
- [x] Code formatting: Run `uvx ruff format packages/samui-backend/ --check`
- [x] Review: Verify all model changes are complete and consistent

**Phase 1 Complete:** Data models updated with ProcessingJob, enhanced ProcessingResult, removed deprecated fields.

---

## Phase 2: Schema Updates

**Goal:** Update Pydantic schemas for new models and removed fields.

**Deliverable:** API schemas matching new data model.

**Tasks:**

- [x] [P2.1] Add `JobStatus` to schema imports in `schemas.py`
- [x] [P2.2] Create `ProcessingJobCreate` schema in `schemas.py`
  - Fields: image_ids (list[UUID]), mode
- [x] [P2.3] Create `ProcessingJobResponse` schema in `schemas.py`
  - Fields: id, mode, status, image_ids, current_index, created_at, started_at, completed_at, error, image_count (computed)
- [x] [P2.4] Create `ProcessingHistoryResponse` schema in `schemas.py`
  - Fields: id, job_id, mode, processed_at, text_prompt_used, bboxes, mask_blob_path
- [x] [P2.5] Modify `ImageResponse` in `schemas.py`: Remove `processing_status` field
- [x] [P2.6] Modify `AnnotationResponse` in `schemas.py`: Remove `source` field if present
- [x] [P2.7] Update any `AnnotationCreate` schema: Remove `source` field if present
- [x] [P2.8] Add `force_all` boolean field to `ProcessingJobCreate` schema (default False)

**Checkpoints:**

- [x] Code quality: Run `uvx ruff check packages/samui-backend/`
- [x] Code formatting: Run `uvx ruff format packages/samui-backend/ --check`
- [x] Review: Verify schemas match model changes

**Phase 2 Complete:** Schemas updated for job-based processing.

---

## Phase 3: Job Processor Service

**Goal:** Create service for job execution and queue management.

**Deliverable:** Working job processor that handles queue and execution.

**Tasks:**

- [x] [P3.1] Create `services/job_processor.py` with `needs_processing(db, image_id, mode)` function
  - Compare current annotation IDs with `latest_result.annotation_ids` (set comparison)
  - Compare text_prompt with text_prompt_used for find-all mode
  - Any difference in annotation sets triggers reprocessing (catches adds and deletes)
- [x] [P3.2] Create `get_images_needing_processing(db, image_ids, mode)` function in job_processor.py
  - Filter image_ids by needs_processing check
  - Return list of image_ids that need processing
- [x] [P3.3] Create `process_single_image(db, storage, sam3, image, job, mode)` function in job_processor.py
  - Run inference, save mask and COCO JSON
  - Create ProcessingResult with job_id, annotation_ids, text_prompt_used, bboxes
- [x] [P3.4] Create `process_job(job_id)` function in job_processor.py
  - Set job status to RUNNING
  - Process each image, update current_index
  - Set job status to COMPLETED or FAILED
- [x] [P3.5] Create `process_job_and_check_queue(job_id)` function in job_processor.py
  - Call process_job
  - Check for next QUEUED job and process it (recursive)
- [x] [P3.6] Create `start_job_if_none_running(db, background_tasks, job_id)` function
  - Check if any job is RUNNING
  - If not, start background task for this job
- [x] [P3.7] Create `tests/test_job_processor.py` with unit tests
  - Test needs_processing logic (all cases)
  - Test job state transitions
- [x] [P3.8] Add startup cleanup function in `job_processor.py`: reset RUNNING jobs to FAILED on app startup
- [x] [P3.9] Register startup cleanup in `main.py` lifespan context manager
- [x] [P3.10] Add `get_background_db()` context manager in `db/database.py` for background task sessions

**Checkpoints:**

- [x] Code quality: Run `uvx ruff check packages/samui-backend/`
- [x] Code formatting: Run `uvx ruff format packages/samui-backend/ --check`
- [x] Tests pass: Run `cd packages/samui-backend && uv run pytest ../../tests/test_job_processor.py -v`

**Phase 3 Complete:** Job processor service implemented with queue management.

---

## Phase 4: Job API Routes

**Goal:** Create API endpoints for job management.

**Deliverable:** Working job CRUD and status endpoints.

**Tasks:**

- [x] [P4.1] Create `routes/jobs.py` with router prefix `/jobs`
- [x] [P4.2] Add `POST /jobs` endpoint to create new processing job
  - Accept image_ids and mode
  - For "Process" button: filter by needs_processing
  - For "Process All": use all provided image_ids
  - Create job, call start_job_if_none_running
- [x] [P4.3] Add `GET /jobs` endpoint to list all jobs
  - Return list of ProcessingJobResponse, newest first
- [x] [P4.4] Add `GET /jobs/{job_id}` endpoint to get job details
  - Return ProcessingJobResponse with current progress
- [x] [P4.5] Add `GET /images/{image_id}/history` endpoint to `routes/images.py`
  - Query parameter: mode
  - Return list of ProcessingHistoryResponse, newest first
- [x] [P4.6] Add `GET /results/{result_id}/mask` endpoint to `routes/jobs.py`
  - Return mask PNG for specific result
- [x] [P4.7] Register jobs router in `main.py`
- [x] [P4.8] Create `tests/test_api_jobs.py` with integration tests
  - Test job creation
  - Test job listing
  - Test job status retrieval
  - Test history retrieval
  - Test mask retrieval (`/results/{result_id}/mask`)
- [x] [P4.9] Run job tests: `cd packages/samui-backend && uv run pytest ../../tests/test_api_jobs.py -v`
- [x] [P4.10] Add validation in `POST /jobs`: verify image_ids exist, return 400 if list is empty

**Checkpoints:**

- [x] Code quality: Run `uvx ruff check packages/samui-backend/`
- [x] Code formatting: Run `uvx ruff format packages/samui-backend/ --check`
- [x] Review: Verify all endpoints work correctly

**Phase 4 Complete:** Job API endpoints implemented and tested.

---

## Phase 5: Update Existing Routes

**Goal:** Update existing routes to remove deprecated fields and use job-based processing.

**Deliverable:** Clean routes without processing_status or source references.

**Tasks:**

- [x] [P5.1] Modify `routes/processing.py`: Remove `_processing_state` dict entirely
- [x] [P5.2] Modify `routes/processing.py`: Remove or update `start_processing` endpoint
  - Either remove (replaced by /jobs POST) or redirect to jobs API
- [x] [P5.3] Modify `routes/processing.py`: Update `get_processing_status` to query current RUNNING job
- [x] [P5.4] Modify `routes/processing.py`: Update mask endpoint to get latest result for image
- [x] [P5.5] Modify `routes/processing.py`: Update COCO export to get latest result for image
- [x] [P5.6] Modify `routes/annotations.py`: Remove code that sets `image.processing_status`
- [x] [P5.7] Modify `routes/annotations.py`: Remove `source` field handling from annotation creation
- [x] [P5.8] Modify `routes/images.py`: Remove `processing_status` from responses
- [x] [P5.9] Update `tests/test_api_processing.py`: Remove processing_status assertions
- [x] [P5.10] Update `tests/test_api_annotations.py`: Remove source field assertions
- [x] [P5.11] Update `tests/test_api_images.py`: Remove processing_status assertions
- [x] [P5.12] Run all backend tests: `cd packages/samui-backend && uv run pytest ../../tests/ -v`

**Checkpoints:**

- [x] Code quality: Run `uvx ruff check packages/samui-backend/`
- [x] Code formatting: Run `uvx ruff format packages/samui-backend/ --check`
- [x] All tests pass

**Phase 5 Complete:** Existing routes updated, deprecated fields removed, all tests pass.

---

## Phase 6: Frontend API Updates

**Goal:** Add frontend API functions for jobs and history.

**Deliverable:** Frontend can interact with new job-based API.

**Tasks:**

- [x] [P6.1] Add `create_job(image_ids, mode, process_all=False)` function to `api.py`
  - POST to /jobs with appropriate parameters
- [x] [P6.2] Add `fetch_jobs()` function to `api.py`
  - GET /jobs, return list of jobs
- [x] [P6.3] Add `fetch_job(job_id)` function to `api.py`
  - GET /jobs/{job_id}, return job details including status fields: `is_running`, `processed_count`, `current_image_filename`
- [x] [P6.4] Add `fetch_image_history(image_id, mode)` function to `api.py`
  - GET /images/{image_id}/history, return list of results
- [x] [P6.5] Add `fetch_result_mask(result_id)` function to `api.py`
  - GET /results/{result_id}/mask, return mask bytes

**Checkpoints:**

- [x] Code quality: Run `uvx ruff check packages/samui-frontend/`
- [x] Code formatting: Run `uvx ruff format packages/samui-frontend/ --check`

**Phase 6 Complete:** Frontend API functions ready for new endpoints.

---

## Phase 7: Processing Page Updates

**Goal:** Update processing page for job-based flow.

**Deliverable:** Processing page with "Process" and "Process All" buttons using job API.

**Tasks:**

- [x] [P7.1] Modify `pages/processing.py`: Remove references to `processing_status` field
- [x] [P7.2] Modify `pages/processing.py`: Add "Process All" button next to "Process" button
- [x] [P7.3] Modify `pages/processing.py`: Update "Process" button to call create_job with process_all=False
- [x] [P7.4] Modify `pages/processing.py`: Update "Process All" button to call create_job with process_all=True
- [x] [P7.5] Modify `pages/processing.py`: Update progress display to poll `GET /jobs/{job_id}` using `fetch_job()`
  - Store job_id from create_job response
  - Use computed fields: `is_running`, `processed_count`, `current_image_filename`
- [x] [P7.6] Modify `pages/annotation.py`: Remove any references to `processing_status`
- [x] [P7.7] Modify `pages/annotation.py`: Remove any references to annotation `source`
- [x] [P7.8] Manual test: Verify processing page works with new job-based flow

**Checkpoints:**

- [x] Code quality: Run `uvx ruff check packages/samui-frontend/`
- [x] Code formatting: Run `uvx ruff format packages/samui-frontend/ --check`
- [x] Review: Processing flow works end-to-end

**Phase 7 Complete:** Processing page updated for job-based flow with both buttons.

---

## Phase 8: Arrow Navigator Component

**Goal:** Create reusable arrow navigation component for image cycling.

**Deliverable:** Working ArrowNavigator component.

**Tasks:**

- [x] [P8.1] Create `components/arrow_navigator.py` with `arrow_navigator(items, key_prefix)` function
  - Display: `< [current/total] >` with left/right buttons
  - Store current index in session state
  - Return current index
- [x] [P8.2] Handle edge cases in arrow_navigator
  - Disable left button at index 0
  - Disable right button at last index
  - Handle empty list and single item
- [x] [P8.3] Manual test: Verify arrow navigation works correctly

**Checkpoints:**

- [x] Code quality: Run `uvx ruff check packages/samui-frontend/`
- [x] Code formatting: Run `uvx ruff format packages/samui-frontend/ --check`

**Phase 8 Complete:** Arrow navigator component ready for use.

---

## Phase 9: Image History Page

**Goal:** Create page showing image with processing history gallery.

**Deliverable:** Working Image History page with mask+bbox overlays.

**Tasks:**

- [ ] [P9.1] Create `pages/history.py` with basic page structure
  - Page title and mode toggle
  - Fetch all images from API
- [ ] [P9.2] Add arrow navigator for image selection
  - Use ArrowNavigator component
  - Display current image filename
- [ ] [P9.3] Display original image at top of page
  - Fetch and display selected image
  - Show filename and dimensions
- [ ] [P9.4] Fetch processing history for selected image
  - Call fetch_image_history(image_id, mode)
- [ ] [P9.5] Create `_create_history_overlay(result, image_data)` function
  - Fetch mask using fetch_result_mask
  - Draw bboxes from result.bboxes
  - Apply mask overlay
  - Return PIL Image
- [ ] [P9.6] Display history gallery with timestamps
  - Use image_gallery component
  - Show processed_at formatted as label
  - Newest first ordering
- [ ] [P9.7] Update `app.py`: Add History page to sidebar navigation
- [ ] [P9.8] Manual test: End-to-end history page flow

**Checkpoints:**

- [ ] Code quality: Run `uvx ruff check packages/samui-frontend/`
- [ ] Code formatting: Run `uvx ruff format packages/samui-frontend/ --check`

**Phase 9 Complete:** Image History page fully functional.

---

## Phase 10: Jobs Page

**Goal:** Create page listing all processing jobs with status.

**Deliverable:** Working Jobs page with status display.

**Tasks:**

- [ ] [P10.1] Create `pages/jobs.py` with basic page structure
  - Page title
  - Fetch all jobs from API
- [ ] [P10.2] Display job list with status indicators
  - Show status icon/badge (queued, running, completed, failed)
  - Show mode, image count, timestamps
- [ ] [P10.3] Display progress for running jobs
  - Show current_index / image_count
  - Show current image filename if available
- [ ] [P10.4] Display duration for completed jobs
  - Calculate from started_at to completed_at
- [ ] [P10.5] Display error message for failed jobs
- [ ] [P10.6] Add auto-refresh for running jobs
  - Poll for updates while any job is running
- [ ] [P10.7] Update `app.py`: Add Jobs page to sidebar navigation
- [ ] [P10.8] Manual test: End-to-end jobs page flow

**Checkpoints:**

- [ ] Code quality: Run `uvx ruff check packages/samui-frontend/`
- [ ] Code formatting: Run `uvx ruff format packages/samui-frontend/ --check`

**Phase 10 Complete:** Jobs page fully functional.

---

## Phase 11: Integration Testing and Cleanup

**Goal:** Verify complete feature works end-to-end.

**Deliverable:** All tests passing, feature complete.

**Tasks:**

- [ ] [P11.1] Run full test suite: `cd packages/samui-backend && uv run pytest ../../tests/ -v`
- [ ] [P11.2] Manual test: Complete workflow
  - Upload images
  - Add annotations
  - Click "Process" - verify only changed images processed
  - Click "Process All" - verify all images processed
  - View Jobs page - verify job status
  - View History page - verify results with mask and bboxes
- [ ] [P11.3] Manual test: Job queue
  - Start processing job
  - Immediately start another job
  - Verify second job is queued
  - Verify second job runs after first completes
- [ ] [P11.4] Manual test: Change detection
  - Process image
  - Modify annotation
  - Process again - verify image is reprocessed
  - Process again without changes - verify image is skipped
- [ ] [P11.5] Clean up any TODO comments or temporary code
- [ ] [P11.6] Run security scan: `uvx bandit -c pyproject.toml -r packages/`
- [ ] [P11.7] Delete old plan files: `rm plans/03-status-*.md plans/04-reprocess-*.md`

**Checkpoints:**

- [ ] Code quality: Run `uvx ruff check packages/`
- [ ] Code formatting: Run `uvx ruff format packages/ --check`
- [ ] All tests pass
- [ ] Review: Final review of all changes

**Phase 11 Complete:** Job-based processing feature complete. Jobs queue and execute FIFO, history is preserved, change detection works via timestamps, new UI pages display jobs and history.
