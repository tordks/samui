# Job-Based Processing Feature Plan

## Overview

### Problem

The current SAM3 WebUI processing system has several limitations:

1. **Status tracking issues:** `Image.processing_status` field can become out of sync with actual data state
2. **In-memory state:** `_processing_state` dict in processing routes is not accessible from other modules
3. **No change detection:** System cannot detect if annotations changed after processing
4. **No history:** Only one `ProcessingResult` per image+mode (unique constraint overwrites previous results)
5. **No job queue:** Cannot queue multiple processing requests
6. **Mixed annotations:** Find-all discoveries stored as `Annotation(source=MODEL)`, mixing user intent with system output

### Purpose

Redesign the processing system around discrete jobs that:
1. Queue and execute one at a time (FIFO)
2. Track their own lifecycle in the database
3. Detect changes via annotation ID set comparison (single source of truth)
4. Store complete history (multiple results per image)
5. Separate user annotations from model discoveries

### Scope

**IN scope:**

- New `ProcessingJob` table with queue support
- Remove unique constraint on `ProcessingResult` to enable history
- Add `annotation_ids`, `bboxes`, and `text_prompt_used` fields to `ProcessingResult`
- Remove `Image.processing_status` field
- Remove `Annotation.source` field (all annotations are user-created)
- Annotation ID set comparison for change detection (derived from annotations, single source of truth)
- Two processing buttons: "Process" (changed only) and "Process All" (force all)
- New Image History page with original image and results gallery
- New Jobs page listing all jobs with status
- Job queue: multiple jobs queued, one runs at a time

**OUT of scope:**

- API/Worker split (documented as future work)
- Celery/RabbitMQ integration (future Azure deployment)
- Automatic retry on failure
- Job cancellation
- Race condition handling for concurrent job starts (queue uses check-then-start pattern; acceptable for single-user)

### Success Criteria

- Jobs queue in database and execute FIFO
- Processing creates new result rows (history preserved)
- Change detection works via annotation ID set comparison
- Image History page shows original image with mask+bbox overlay gallery
- Jobs page shows all jobs with status (queued, running, completed, failed)
- MODEL annotations removed; discoveries stored in `ProcessingResult.bboxes`
- All existing tests pass after refactor

---

## Solution Design

### System Architecture

**Core Components:**

- **ProcessingJob:** New table tracking batch processing jobs with queue support
- **ProcessingResult (modified):** Stores all results with `bboxes` and `text_prompt_used`
- **JobProcessor:** Service handling job execution and queue management
- **Image History Page:** New UI showing image with processing history gallery
- **Jobs Page:** New UI listing all jobs with status

**Project Structure:**

```
packages/samui-backend/src/samui_backend/
├── db/
│   └── models.py [MODIFY] - Add ProcessingJob, update ProcessingResult, remove Image.processing_status
├── enums.py [MODIFY] - Add JobStatus enum, remove AnnotationSource
├── routes/
│   ├── processing.py [MODIFY] - Rewrite for job-based flow with queue
│   ├── jobs.py [CREATE] - Job CRUD and status endpoints
│   ├── annotations.py [MODIFY] - Remove source field handling
│   └── images.py [MODIFY] - Remove processing_status from responses
├── services/
│   └── job_processor.py [CREATE] - Job execution and queue logic
└── schemas.py [MODIFY] - Add job schemas, update result schemas

packages/samui-frontend/src/samui_frontend/
├── api.py [MODIFY] - Add job and history API calls
├── components/
│   └── arrow_navigator.py [CREATE] - Reusable image navigation
├── pages/
│   ├── history.py [CREATE] - Image History page
│   ├── jobs.py [CREATE] - Jobs list page
│   └── processing.py [MODIFY] - Add "Process All" button, use job API
└── app.py [MODIFY] - Add new pages to navigation

tests/
├── test_api_jobs.py [CREATE] - Job endpoint tests
├── test_job_processor.py [CREATE] - Job execution tests
├── test_api_processing.py [MODIFY] - Update for job-based flow
├── test_api_annotations.py [MODIFY] - Remove source field tests
└── test_api_images.py [MODIFY] - Remove processing_status tests
```

**Component Relationships:**

- Routes create jobs via `ProcessingJob` model
- `JobProcessor` service handles execution and queue management
- Background task calls `JobProcessor` to process jobs
- `ProcessingResult` references `ProcessingJob` via `job_id`
- Frontend polls job status and fetches results

**Relationship to Existing Codebase:**

- Replaces current `_processing_state` in-memory dict with database state
- Removes `Image.processing_status` field (derived from job/result data)
- Removes `Annotation.source` field (all annotations are user-created)
- Follows existing service pattern in `services/` directory
- Uses existing `image_gallery` component for history display

---

### Design Rationale

**Job-based processing instead of per-image status**

Jobs are first-class entities representing discrete batches of work. This eliminates status sync issues - job status is authoritative and stored in database.

Alternatives considered:
- Fix status sync logic: Complex, error-prone, still requires in-memory state
- Per-mode status fields on Image: Hardcodes modes, sync issues remain

Trade-offs accepted:
- Pro: Single source of truth, no sync bugs
- Pro: Natural history via job_id grouping
- Con: Additional table and complexity

**Annotation ID set comparison for change detection**

Store the IDs of annotations used for each processing result. Change detection compares current annotation IDs with the stored set - any difference (addition or deletion) triggers reprocessing.

For text prompts: store `text_prompt_used` snapshot in result, compare with current `image.text_prompt`.

Alternatives considered:
- Timestamp-based: Doesn't detect deletions without also tracking count
- Separate AnnotationState table: Introduces sync/race condition risks
- Hash-based detection: More complex, requires hash computation

Trade-offs accepted:
- Pro: Single source of truth (annotations table is authoritative)
- Pro: Exact change detection (catches adds, deletes, any difference)
- Pro: Result stores exactly what was processed (useful for debugging)
- Con: Set comparison is O(n) but negligible for typical annotation counts

**Remove MODEL annotations, store discoveries in results**

Find-all mode discoveries stored in `ProcessingResult.bboxes` JSON field instead of `Annotation` rows. Simplifies annotation model - all annotations represent user intent.

Alternatives considered:
- Keep MODEL annotations: Mixing intent with output, complex queries
- Separate discoveries table: More complexity for little benefit

Trade-offs accepted:
- Pro: Clean annotation model
- Pro: Discoveries naturally tied to specific processing result
- Con: Cannot edit discoveries (acceptable - they're output, not input)

**Database-backed job queue with BackgroundTasks**

Jobs queue in database, FastAPI BackgroundTasks handles execution. One job runs at a time, next job starts when current completes.

Alternatives considered:
- Celery/RabbitMQ: Overkill for single-user tool, adds infrastructure
- No queue: Poor UX when processing takes time

Trade-offs accepted:
- Pro: Simple, no external dependencies
- Pro: Easy migration path to Azure Queue later
- Con: Queue state lost on process crash (acceptable for single-user)

---

### Technical Specification

**Dependencies:**

Required libraries (existing):
- SQLAlchemy 2.0+ (database ORM)
- FastAPI 0.100+ (API framework)
- Pydantic 2.0+ (request/response models)
- Streamlit 1.30+ (frontend framework)

No new dependencies required.

**Data Model:**

```python
class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class ProcessingJob(Base):
    id: UUID
    mode: SegmentationMode
    status: JobStatus
    image_ids: JSON  # list of UUID strings
    current_index: int  # progress through image_ids
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    error: str | None

class ProcessingResult(Base):
    id: UUID
    job_id: UUID  # FK to ProcessingJob
    image_id: UUID
    mode: SegmentationMode
    processed_at: datetime
    annotation_ids: JSON  # list of UUID strings (input annotations used)
    text_prompt_used: str | None  # snapshot for change detection
    bboxes: JSON  # list of {x, y, width, height} (SAM3 discovered boxes)
    mask_blob_path: str  # masks/{result_id}.png (unique per result for history)
    coco_json_blob_path: str  # coco/{result_id}.json (unique per result for history)
```

**Blob Path Strategy:**

Blob paths use `result_id` instead of `image_id` to preserve history:
- **Old (overwrites):** `masks/{image_id}.png`, `coco/{image_id}.json`
- **New (preserves history):** `masks/{result_id}.png`, `coco/{result_id}.json`

Each ProcessingResult has unique blob files. Old results remain accessible for the history gallery.

**Job Queue Flow:**

```
POST /jobs
    │
    ▼
Create ProcessingJob (status=QUEUED)
    │
    ▼
Any job RUNNING? ──YES──> Return job_id (queued)
    │
    NO
    │
    ▼
Start BackgroundTask
    │
    ▼
Process job (status=RUNNING)
    │
    ▼
For each image:
    - Create ProcessingResult
    - Update current_index
    │
    ▼
Job complete (status=COMPLETED)
    │
    ▼
Next QUEUED job exists? ──YES──> Process next job
    │
    NO
    │
    ▼
Worker idle
```

**Change Detection:**

```python
def needs_processing(db, image_id, mode) -> bool:
    """Derive processing need by comparing current annotations vs last result."""
    image = db.get(Image, image_id)
    annotations = get_annotations_for_mode(db, image_id, mode)
    current_ids = {str(a.id) for a in annotations}

    latest_result = get_latest_result(db, image_id, mode)

    # No previous result - process if there's something to process
    if not latest_result:
        if current_ids:
            return True
        # Find-all can run with just text prompt
        if mode == FIND_ALL and image.text_prompt:
            return True
        return False

    # Compare annotation sets (catches adds AND deletes)
    processed_ids = set(latest_result.annotation_ids or [])
    if current_ids != processed_ids:
        return True

    # Check text_prompt changed (find-all mode)
    if mode == FIND_ALL:
        if image.text_prompt != latest_result.text_prompt_used:
            return True

    return False
```

**API Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/jobs` | POST | Create new processing job |
| `/jobs` | GET | List all jobs |
| `/jobs/{id}` | GET | Get job details and progress |
| `/images/{id}/history` | GET | Get processing history for image |
| `/results/{id}/mask` | GET | Get mask for specific result |

**Error Handling:**

- Job failure: Set status=FAILED, store error message, continue to next queued job
- Individual image failure: Log error, continue to next image in job
- Database error: Log, job may be in inconsistent state (acceptable for single-user)

**Startup Cleanup:**

On application startup, reset any jobs in RUNNING state to FAILED. This handles the case where the server crashed or restarted while a job was in progress. The job processor lifespan or startup event calls cleanup logic before accepting new requests.

**Background Task Session Management:**

Background tasks run after the HTTP response is sent, so the request-scoped database session is closed. Background tasks must create their own session using a context manager:

```python
# db/database.py
@contextmanager
def get_background_db():
    """Database session for background tasks (not request-scoped)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# services/job_processor.py
def process_job(job_id: UUID):
    with get_background_db() as db:
        job = db.get(ProcessingJob, job_id)
        # ... process job ...
```

This ensures proper session lifecycle and enables easy mocking in tests.

---

## Implementation Strategy

### Development Approach

**Backend-first with incremental migration:**

1. **Models first:** Add new tables/fields before changing logic
2. **Services next:** Create job processor service
3. **Routes:** Migrate processing routes to job-based flow
4. **Cleanup:** Remove old fields and code
5. **Frontend last:** Update UI after backend is stable

This order minimizes risk - each step is independently testable.

### Testing Approach

**Regression-focused with new integration tests:**

- Ensure existing annotation/image tests pass after field removals
- Add tests for job lifecycle (create, queue, execute, complete)
- Add tests for change detection logic
- Add tests for history retrieval
- Manual testing for UI pages

### Risk Mitigation

- **Queue state on crash:** Acceptable for single-user; queued jobs remain in DB
- **Concurrent job start:** Check for RUNNING job before starting new one
- **Large job handling:** Process images one at a time, update progress incrementally

### Checkpoint Strategy

Each phase ends with mandatory validation:

- **Self-review:** Agent reviews implementation against phase deliverable
- **Code quality:** Run `uvx ruff check packages/` and `uvx ruff format packages/ --check`
- **Tests pass:** Run `cd packages/samui-backend && uv run pytest ../../tests/ -v`

---

## Future Work

### API/Worker Split

For Azure multi-user deployment, split into separate services:

```
API Service (lightweight)     Worker Service (GPU-heavy)
├── CRUD endpoints            ├── Polls DB for jobs
├── Job creation              ├── SAM3 model loaded
├── Status queries            ├── Inference
└── No GPU needed             └── Result storage
```

Migration path:
1. Current: Worker logic in `services/job_processor.py`, called via BackgroundTasks
2. Preparation: Switch to database polling pattern
3. Split: Extract worker to separate container
