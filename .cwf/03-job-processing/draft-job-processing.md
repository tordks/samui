# Job-Based Processing Redesign - Analysis Draft

## Background

This document captures analysis and design discussion for redesigning the processing system from per-image status tracking to a job-based approach.

**Context:** The current system tracks processing status per image (`Image.processing_status`) and uses in-memory state (`_processing_state`) for progress tracking. This creates sync issues and complexity. The proposed approach simplifies by treating processing as discrete jobs.

---

## Current System (Problems)

```
Image
├── processing_status: PENDING | ANNOTATED | PROCESSING | PROCESSED
├── annotations[]
│   └── source: USER | MODEL  ← MODEL for find-all discoveries
└── processing_result (unique per image+mode)

_processing_state (in-memory)
├── is_running
├── current_image_id
├── processed_count
└── total_count
```

**Issues:**
1. `processing_status` can become out of sync with actual data
2. In-memory state not accessible from other modules
3. No change detection (always shows PROCESSED even if annotations changed)
4. MODEL annotations mix user intent with system output
5. Unique constraint prevents storing processing history

---

## Proposed System (Job-Based)

### Core Principles

1. **Jobs are first-class entities** - A ProcessingJob represents a discrete batch of work
2. **Results store everything** - Mask, bboxes, COCO JSON all in ProcessingResult
3. **Timestamp-based change detection** - Compare annotation timestamps vs result timestamps
4. **No per-image status** - Status derived from job/result existence and timestamps
5. **Annotations are user-only** - Remove MODEL source; find-all discoveries go into results

### Data Model

```python
class ProcessingJob(Base):
    """A batch processing job."""
    id: UUID
    mode: SegmentationMode
    status: JobStatus  # pending, running, completed, failed
    image_ids: JSON  # list of UUID strings - images to process
    current_index: int  # progress through image_ids
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    error: str | None

    # Relationships
    results: list[ProcessingResult]

    # Derived properties
    @property
    def image_count(self) -> int:
        return len(self.image_ids)

    @property
    def processed_count(self) -> int:
        return self.current_index

class ProcessingResult(Base):
    """Result of processing a single image within a job."""
    id: UUID
    job_id: UUID  # FK to ProcessingJob
    image_id: UUID  # FK to Image
    mode: SegmentationMode
    processed_at: datetime

    # Input snapshot (for change detection)
    text_prompt_used: str | None  # Snapshot of image.text_prompt at processing time

    # Output data
    mask_blob_path: str
    coco_json_blob_path: str
    bboxes: JSON  # list of {x, y, width, height} dicts

    # Relationships
    job: ProcessingJob
    image: Image

class Annotation(Base):
    """User-created bounding box annotation (immutable - create or delete only)."""
    id: UUID
    image_id: UUID
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    prompt_type: PromptType  # SEGMENT | POSITIVE_EXEMPLAR | NEGATIVE_EXEMPLAR
    created_at: datetime  # Used for change detection (newest annotation vs latest result)

    # REMOVED: source field (was USER | MODEL)
    # NOTE: No updated_at - annotations are immutable (delete and recreate to change)
```

### Removed Fields

| Field | Reason for Removal |
|-------|-------------------|
| `Image.processing_status` | Derived from job/result existence and timestamps |
| `Annotation.source` | All annotations are user-created; discoveries go in results |
| `_processing_state` dict | Job status stored in database |

### New Fields

| Field | Purpose |
|-------|---------|
| `ProcessingJob.image_ids` | JSON list of image UUIDs to process |
| `ProcessingJob.current_index` | Progress through image_ids list |
| `ProcessingResult.text_prompt_used` | Snapshot of text_prompt at processing time (for change detection) |
| `ProcessingResult.bboxes` | Store discovered/processed bounding boxes (JSON) |

---

## Change Detection Logic

Instead of hash-based comparison, use simple timestamp comparison.

### Annotation Workflow

**Key insight:** Annotations are immutable - they are created or deleted, never updated.

- To "change" an annotation: delete old one, create new one
- Each annotation has `created_at` timestamp (existing field)
- No `updated_at` field needed

### Text Prompt Tracking

For find-all mode, we need to detect text_prompt changes. Store a snapshot in ProcessingResult:

```python
class ProcessingResult:
    text_prompt_used: str | None  # Snapshot of text_prompt at processing time
```

This allows comparing `image.text_prompt != latest_result.text_prompt_used` to detect changes.

### Change Detection Implementation

```python
def needs_processing(image_id: UUID, mode: SegmentationMode, db: Session) -> bool:
    """Determine if image needs processing based on timestamps and text_prompt."""
    annotations = get_annotations_for_mode(db, image_id, mode)
    image = get_image(db, image_id)

    # Check if there's anything to process
    if mode == SegmentationMode.INSIDE_BOX:
        if not annotations:
            return False
    else:  # FIND_ALL
        if not annotations and not image.text_prompt:
            return False

    # Get latest completed result
    latest_result = (
        db.query(ProcessingResult)
        .filter(
            ProcessingResult.image_id == image_id,
            ProcessingResult.mode == mode,
        )
        .order_by(ProcessingResult.processed_at.desc())
        .first()
    )

    if not latest_result:
        return True  # Never processed

    # Check if any annotation is newer than the result
    if annotations:
        newest_annotation = max(a.created_at for a in annotations)
        if newest_annotation > latest_result.processed_at:
            return True

    # Check if text_prompt changed (find-all mode)
    if mode == SegmentationMode.FIND_ALL:
        if image.text_prompt != latest_result.text_prompt_used:
            return True

    return False
```

**Advantages over hash-based:**
- Simpler implementation (no hash computation)
- No cache needed
- Easy to understand and debug
- Uses existing `created_at` field on Annotation
- Preserves text_prompt history in results

---

## UI: Two Processing Buttons

```
┌─────────────────────────────────────────────────────┐
│  Processing Page                                    │
│                                                     │
│  [Process]  [Process All]                           │
│      ↓           ↓                                  │
│   Changed     All with                              │
│   images      annotations                           │
│   only        (force)                               │
└─────────────────────────────────────────────────────┘
```

### "Process" Button (Default)
- Filters images where `annotation.updated_at > latest_result.processed_at`
- Skips images that haven't changed since last processing
- Most common use case

### "Process All" Button (Force)
- Includes ALL images with annotations/text_prompt for current mode
- Ignores timestamps
- Use case: regenerate all results, testing, etc.

---

## Processing Flow

```
User clicks "Process" or "Process All"
              ↓
┌─────────────────────────────────────────┐
│ 1. Determine images to process          │
│    - "Process": filter by timestamps    │
│    - "Process All": all with annotations│
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 2. Create ProcessingJob                 │
│    - status = 'queued'                  │
│    - image_count = len(images)          │
│    - Store image_ids (how? see below)   │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 3. Background task picks up job         │
│    - status = 'running'                 │
│    - started_at = now()                 │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 4. For each image:                      │
│    - Load image and annotations         │
│    - Run SAM3 inference                 │
│    - Create ProcessingResult            │
│      - Store mask, COCO JSON, bboxes    │
│    - Increment processed_count          │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 5. Job completes                        │
│    - status = 'completed' or 'failed'   │
│    - completed_at = now()               │
└─────────────────────────────────────────┘
```

### Open Question: How to store images-to-process in job?

**Option A: Separate join table**
```python
class ProcessingJobImage(Base):
    job_id: UUID
    image_id: UUID
```

**Option B: JSON field on job**
```python
class ProcessingJob:
    image_ids: JSON  # list of UUID strings
```

**Option C: Derive from results**
- Don't store upfront; job processes images and creates results
- `image_count` set at creation, results accumulate

Option C is simplest but loses the "which images were intended" if job fails partway.

---

## Concurrent Jobs

**Policy:** Allow multiple concurrent jobs, including on the same image.

- No locking or checking for "is this image in another job"
- If user starts two jobs with overlapping images, both run
- Both create results (history preserved)
- Simple, no coordination needed

**Why this is acceptable:**
- Single-user tool
- User controls when to process
- Redundant processing is harmless (just uses resources)
- Results are append-only (history)

---

## Find-All Mode: Discoveries

Currently, find-all mode creates `Annotation(source=MODEL)` for discovered objects.

**New approach:** Store discoveries only in ProcessingResult.

```python
class ProcessingResult:
    bboxes: JSON  # [{"x": 10, "y": 20, "width": 100, "height": 150}, ...]
```

**Implications:**
- Discoveries are output, not input for further refinement
- Cannot edit discovered boxes (they're not annotations)
- Simplifies annotation model (all annotations are user intent)
- COCO JSON still has full segmentation data

**Trade-off:** Lost editability of discoveries. Acceptable if find-all is primarily for batch discovery, not interactive refinement.

---

## What This Removes

1. **`Image.processing_status`** - No per-image status field
2. **`Annotation.source`** - No USER/MODEL distinction
3. **`_processing_state` dict** - No in-memory state
4. **Hash computation** - No input hashing
5. **Status derivation service** - No complex status logic
6. **LRU cache for status** - Not needed

---

## What This Adds

1. **`ProcessingJob` table** - Job lifecycle tracking with queue support
2. **`ProcessingResult.text_prompt_used`** - Snapshot for change detection
3. **`ProcessingResult.bboxes`** - Store discovered boxes
4. **"Process All" button** - Force reprocessing option
5. **Job queue** - Multiple jobs can be queued, processed FIFO

---

## New UI Pages

### Image History Page

Displays processing history for a single image.

```
┌─────────────────────────────────────────────────────────────┐
│  Image History                           Mode: [Inside Box ▼]│
│─────────────────────────────────────────────────────────────│
│                                                              │
│  ◄ 3/10 ►   image_003.jpg                                   │
│                                                              │
│  ┌────────────────────────────────────┐                     │
│  │                                    │                     │
│  │         Original Image             │                     │
│  │         (1920 x 1080)              │                     │
│  │                                    │                     │
│  └────────────────────────────────────┘                     │
│                                                              │
│  Processing History (3 results)                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │          │  │          │  │          │                   │
│  │ Result 1 │  │ Result 2 │  │ Result 3 │                   │
│  │ + mask   │  │ + mask   │  │ + mask   │                   │
│  │          │  │          │  │          │                   │
│  ├──────────┤  ├──────────┤  ├──────────┤                   │
│  │2024-01-10│  │2024-01-11│  │2024-01-13│                   │
│  │ 14:30    │  │ 09:15    │  │ 11:45    │                   │
│  └──────────┘  └──────────┘  └──────────┘                   │
│                              (newest)                        │
└─────────────────────────────────────────────────────────────┘
```

**Features:**
- Arrow navigation to cycle through images
- Mode toggle (Inside Box / Find All)
- Original image display
- Gallery of processing results with mask overlay AND bboxes drawn
- Timestamps on each result

### Jobs Page

Lists all processing jobs with status.

```
┌─────────────────────────────────────────────────────────────┐
│  Processing Jobs                                             │
│─────────────────────────────────────────────────────────────│
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ ● RUNNING    Job #4       Inside Box    3/5 images      ││
│  │              Started: 2024-01-13 11:42                  ││
│  │              Currently: image_007.jpg                   ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ ○ QUEUED     Job #5       Find All      10 images       ││
│  │              Created: 2024-01-13 11:43                  ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ ✓ COMPLETED  Job #3       Inside Box    8/8 images      ││
│  │              Completed: 2024-01-13 11:40 (2m 34s)       ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ ✗ FAILED     Job #2       Find All      3/5 images      ││
│  │              Failed: 2024-01-13 10:15                   ││
│  │              Error: CUDA out of memory                  ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Features:**
- List all jobs, newest first
- Status indicators (queued, running, completed, failed)
- Progress for running jobs (X/Y images)
- Duration for completed jobs
- Error message for failed jobs
- Mode indicator (Inside Box / Find All)

---

## Execution Architecture

### Current Implementation: FastAPI BackgroundTasks

```
┌─────────────────────────────────────────────────────────────┐
│ FastAPI Process                                             │
│                                                             │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐ │
│  │ POST        │      │ Background  │      │ Database    │ │
│  │ /process    │─────>│ Task        │─────>│ Job Table   │ │
│  │             │      │ (in-process)│      │             │ │
│  └─────────────┘      └─────────────┘      └─────────────┘ │
│                              │                              │
│                              ▼                              │
│                       ┌─────────────┐                       │
│                       │ SAM3 GPU    │                       │
│                       │ Inference   │                       │
│                       └─────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

**Flow:**
1. API receives request, creates `ProcessingJob` with status=`queued`
2. Checks if any job is currently `running`
3. If no job running: starts background task for this job
4. If job running: returns immediately (job is queued)
5. Background task processes job, then checks for next queued job
6. Client polls `/jobs/{job_id}` for progress

**Queue logic:**
```python
@router.post("/process")
def start_processing(request: ProcessRequest, background_tasks: BackgroundTasks, db: Session):
    # Create job with queued status
    job = ProcessingJob(status=JobStatus.QUEUED, image_ids=request.image_ids, ...)
    db.add(job)
    db.commit()

    # Check if any job is running
    running_job = db.query(ProcessingJob).filter(
        ProcessingJob.status == JobStatus.RUNNING
    ).first()

    if not running_job:
        # No job running, start this one
        background_tasks.add_task(process_job_and_check_queue, job.id)

    return {"job_id": job.id, "status": "queued"}

def process_job_and_check_queue(job_id: UUID):
    """Process job, then check for next queued job."""
    db = SessionLocal()
    try:
        process_single_job(db, job_id)
    finally:
        # Check for next queued job
        next_job = db.query(ProcessingJob).filter(
            ProcessingJob.status == JobStatus.QUEUED
        ).order_by(ProcessingJob.created_at).first()

        if next_job:
            process_job_and_check_queue(next_job.id)  # Process next

        db.close()
```

**Characteristics:**
- Jobs queue up in database
- One job runs at a time (GPU constraint)
- Next job starts automatically when current finishes
- Queue order: first-in-first-out (by `created_at`)

**Limitations (acceptable for now):**
- Job lost if process crashes mid-execution
- Queue state lost on API restart (queued jobs remain in DB, but worker stops)

### Future Migration: Azure Queue + Worker

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│ FastAPI API │      │ Azure Queue │      │ Worker(s)   │
│             │─────>│ Storage     │─────>│ (separate   │
│ POST        │      │             │      │  container) │
│ /process    │      │ job messages│      │             │
└─────────────┘      └─────────────┘      └─────────────┘
       │                                         │
       │                                         ▼
       │              ┌─────────────┐      ┌─────────────┐
       └─────────────>│ Database    │<─────│ SAM3 GPU    │
                      │ Job Table   │      │ Inference   │
                      └─────────────┘      └─────────────┘
```

**Migration path:**
1. Abstract job dispatch behind interface
2. Replace `background_tasks.add_task()` with queue message
3. Deploy separate worker container(s)
4. Scale workers based on load

**Code structure to enable migration:**

```python
# services/job_dispatcher.py
class JobDispatcher(Protocol):
    def dispatch(self, job_id: UUID) -> None: ...

class BackgroundTaskDispatcher:
    """Current: runs in-process"""
    def __init__(self, background_tasks: BackgroundTasks):
        self.background_tasks = background_tasks

    def dispatch(self, job_id: UUID) -> None:
        self.background_tasks.add_task(process_job, job_id)

class AzureQueueDispatcher:
    """Future: sends to Azure Queue"""
    def __init__(self, queue_client: QueueClient):
        self.queue_client = queue_client

    def dispatch(self, job_id: UUID) -> None:
        self.queue_client.send_message(str(job_id))
```

This abstraction allows swapping dispatch mechanism without changing route logic.

---

## Comparison: Previous Plans vs Job Approach

| Aspect | 03-status + 04-reprocess | Job-based |
|--------|--------------------------|-----------|
| Status tracking | Per-image field + derivation | Job-level only |
| Change detection | Hash-based (SHA-256) | Timestamp-based |
| Duplicate prevention | Automatic (hash match) | User choice (two buttons) |
| MODEL annotations | Kept | Removed |
| Complexity | ~12 phases, ~80 tasks | ~5-6 phases, ~40 tasks |
| New dependencies | cachetools | None |

---

## Resolved Questions

1. **~~`updated_at` on Annotation~~** - Not needed. Annotations are immutable (create/delete only). Use existing `created_at` for change detection.

2. **Job-to-image relationship** - Use JSON field (`image_ids: list[UUID]`). Simple, records intent, enables restart. No need for join table complexity.

3. **Failed job handling** - User starts new job. No automatic retry for single-user tool.

4. **Task execution model** - FastAPI BackgroundTasks with database job tracking and queue. Simple for now, migrate to Celery/Azure Queue later for multi-user.

5. **UI pages** - Two new pages:
   - **Image History page** - Original image + gallery of processed results below
   - **Jobs page** - List of all jobs with status (queued, running, completed, failed)

---

## Future Work: Architecture Split

For Azure multi-user deployment, consider splitting into separate API and Worker services.

### Target Architecture

```
┌─────────────────────┐    ┌─────────────────────┐
│ API Service         │    │ Worker Service      │
│ (lightweight)       │    │ (GPU-heavy)         │
│                     │    │                     │
│ • CRUD endpoints    │    │ • Polls DB for jobs │
│ • Job creation      │    │ • SAM3 model loaded │
│ • Status queries    │    │ • Inference         │
│ • No GPU needed     │    │ • Result storage    │
│                     │    │                     │
└────────┬────────────┘    └──────────┬──────────┘
         │                            │
         └──────────┬─────────────────┘
                    ▼
         ┌─────────────────────┐
         │ Database + Storage  │
         └─────────────────────┘
```

### Benefits

- **Resource isolation:** API container small (256MB), Worker container large (8GB+, GPU)
- **Failure isolation:** Worker crash doesn't affect API - users can still browse/annotate
- **Scaling:** Scale API for users, workers for processing throughput
- **Azure fit:** Natural Container Apps pattern with separate API and worker containers

### Migration Path

1. **Current (monolith):** Worker logic in `services/worker.py`, called via BackgroundTasks
2. **Preparation:** Switch from BackgroundTasks to database polling (worker loop)
3. **Split:** Extract worker to separate service/container, minimal code changes

### Worker Polling Pattern (Future)

```python
# services/worker.py
def run_worker_loop():
    """Poll for jobs and process them. Runs as separate service."""
    sam3 = SAM3Service()
    sam3.load_model()  # Load once, keep loaded

    while True:
        job = get_next_queued_job()  # Poll database
        if job:
            process_job(job, sam3)
        else:
            time.sleep(1)  # Wait before polling again
```

This pattern keeps model loaded (faster inference) and works both in-process and as separate service.

---

## Next Steps

1. Convert this draft to formal plan (`03-job-processing-plan.md`)
2. Create tasklist (`03-job-processing-tasklist.md`)
3. Delete old `03-status-*` and `04-reprocess-*` files
