# SAM3 WebUI Plan

## Overview

### Problem

Segmenting images with SAM3 currently requires writing Python code and manually managing inputs/outputs. There is no visual interface for uploading images, drawing bounding box prompts, running inference, and exporting results in standard formats. This limits accessibility for users who need segmentation masks but lack ML engineering expertise.

### Purpose

Build a web-based UI for SAM3 image segmentation that allows users to:
1. Upload images via drag-and-drop or folder selection
2. Annotate images by drawing bounding boxes as prompts
3. Process annotated images through SAM3 to generate segmentation masks
4. Export results in COCO annotation format (one JSON per image)

### Scope

**IN scope:**
- Single-user local tool
- Image upload with drag-and-drop and folder selection
- Bounding box annotation (positive prompts only)
- Batch processing of annotated images via SAM3
- Progress tracking during processing (per-image DB updates)
- COCO JSON export (one file per image containing all annotations)
- Docker Compose deployment with PostgreSQL and Azurite

**OUT of scope:**
- Multi-user authentication and workspace isolation
- Point prompts (positive/negative clicks)
- Text prompts for SAM3
- Real-time/streaming inference
- Video segmentation
- Cloud deployment (Azure production setup)
- Live bbox preview while drawing (something like use `streamlit-drawable-canvas` (unmaintained) or `streamlit-image-annotation` instead of `streamlit-image-coordinates` for real-time visual feedback during drag)
- Bbox editing (current approach is delete and redraw; could add form inputs for x/y/width/height with +/- adjustment buttons in the annotation sidebar, requires `PUT /annotations/{id}` endpoint)

### Success Criteria

- Users can upload images and view them in a tiled gallery
- Users can draw multiple bounding boxes on images with visual feedback
- Processing runs SAM3 inference on all annotated images with progress indication
- Users can download COCO-format JSON annotations per image
- All components run via `docker compose up`

### UI Reference

See `plans/sam3ui-mockup.html` for interactive mockup of all three pages (Upload, Annotation, Processing).

---

## Solution Design

### System Architecture

**Core Components:**

- **Frontend (Streamlit)**: Web UI with three pages (Upload, Annotation, Processing)
- **Backend (FastAPI)**: REST API handling image storage, annotations, and SAM3 inference
- **SAM3 Service**: Wrapper around SAM3 model for bbox-prompted segmentation
- **Storage Service**: Azure Blob Storage client (Azurite for local dev)
- **Database**: PostgreSQL with SQLAlchemy models, auto-initialized on backend startup

**Project Structure:**
```
samui/
├── pyproject.toml [CREATE]              # uv workspace root
├── docker-compose.yaml [CREATE]
├── .env.example [CREATE]
├── packages/
│   ├── samui-backend/                   # FastAPI + SAM3 (heavy deps)
│   │   ├── pyproject.toml [CREATE]
│   │   ├── Dockerfile [CREATE]
│   │   └── src/
│   │       └── samui_backend/
│   │           ├── __init__.py [CREATE]
│   │           ├── config.py [CREATE]
│   │           ├── main.py [CREATE]
│   │           ├── routes/
│   │           │   ├── __init__.py [CREATE]
│   │           │   ├── images.py [CREATE]
│   │           │   ├── annotations.py [CREATE]
│   │           │   └── processing.py [CREATE]
│   │           ├── schemas.py [CREATE]
│   │           ├── db/
│   │           │   ├── __init__.py [CREATE]
│   │           │   ├── database.py [CREATE]
│   │           │   └── models.py [CREATE]
│   │           └── services/
│   │               ├── __init__.py [CREATE]
│   │               ├── storage.py [CREATE]
│   │               ├── sam3_inference.py [CREATE]
│   │               └── coco_export.py [CREATE]
│   └── samui-frontend/                  # Streamlit only (light deps)
│       ├── pyproject.toml [CREATE]
│       ├── Dockerfile [CREATE]
│       └── src/
│           └── samui_frontend/
│               ├── __init__.py [CREATE]
│               ├── config.py [CREATE]
│               ├── app.py [CREATE]
│               ├── pages/
│               │   ├── __init__.py [CREATE]
│               │   ├── upload.py [CREATE]
│               │   ├── annotation.py [CREATE]
│               │   └── processing.py [CREATE]
│               └── components/
│                   ├── __init__.py [CREATE]
│                   ├── image_gallery.py [CREATE]
│                   └── bbox_annotator.py [CREATE]
└── tests/
    ├── __init__.py [CREATE]
    ├── conftest.py [CREATE]
    ├── test_api_images.py [CREATE]
    ├── test_sam3_inference.py [CREATE]
    └── test_coco_export.py [CREATE]
```

**Component Relationships:**

- Frontend calls Backend endpoints for all data operations
- Backend routes use Storage Service for blob operations
- Backend routes use Database for metadata persistence
- Backend auto-creates DB tables on startup via SQLAlchemy `create_all()`
- Processing route loads SAM3 model, iterates images, updates DB per image
- COCO Export service generates JSON from masks and annotations

**Data Flow:**

```
Startup: Backend → SQLAlchemy create_all() → DB tables ready
Upload: Frontend → POST /images → Storage (blob) + DB (metadata)
Annotate: Frontend → POST /annotations → DB (bbox coords)
Process: Frontend → POST /process → BackgroundTask → SAM3 → Storage (masks) + DB (status)
Export: Frontend → GET /export/{image_id} → COCO JSON
```

---

### Design Rationale

**Streamlit + FastAPI separation**

Streamlit handles UI rendering; FastAPI handles data and inference. This separation allows:
- Non-blocking inference (FastAPI BackgroundTasks)
- Clean API for potential future clients
- Testable backend independent of UI

Alternative considered: Streamlit-only with direct SAM3 calls. Rejected because inference blocks UI and makes progress tracking difficult.

**No task queue (Celery/RQ)**

BackgroundTasks sufficient for single-user batch processing. Adding a task queue introduces Redis dependency and operational complexity without proportional benefit for v1.

Trade-off accepted: If FastAPI crashes mid-batch, processing restarts from scratch. Acceptable for single-user local tool.

**Load SAM3 per batch**

Model loaded when processing starts, released after batch completes. Frees GPU memory when idle.

Alternative considered: Load at startup. Rejected to avoid holding GPU memory during upload/annotation phases.

**COCO JSON per image**

Each image gets its own JSON file containing all annotations. Simpler than single monolithic file; easier to manage partial exports and re-processing.

**streamlit-image-coordinates for bbox annotation**

Third-party component (https://github.com/blackary/streamlit-image-coordinates) provides rectangle selection. Avoids building custom canvas from scratch.

**Auto-initialize DB on startup**

Backend calls `Base.metadata.create_all()` on startup. No separate init script needed - tables are created automatically if they don't exist. Simplifies deployment (just `docker compose up`).

---

### Technical Specification

**Dependencies:**

**samui-backend package** (heavy, ~2GB+ with torch):
- `sam3` (Facebook Research SAM3 package)
- `torch` >= 2.0
- `fastapi` >= 0.100
- `uvicorn` (ASGI server)
- `sqlalchemy` >= 2.0
- `psycopg2-binary` (PostgreSQL driver)
- `azure-storage-blob` (Azurite/Azure Blob Storage client)
- `pydantic` >= 2.0
- `pillow` (image handling)
- `numpy`

**samui-frontend package** (light, no ML deps):
- `streamlit` >= 1.30
- `streamlit-image-coordinates` (bbox drawing component)
- `httpx` (API client)
- `pydantic` >= 2.0
- `pillow` (image display)

**Dev dependencies** (workspace root):
- `pytest`
- `httpx` (async HTTP client for tests)
- `ruff` (linting)

External systems:
- PostgreSQL (via Docker)
- Azurite (Azure Blob Storage emulator, via Docker)

**Database Schema:**

```
images
├── id: UUID (PK)
├── filename: VARCHAR
├── blob_path: VARCHAR
├── width: INTEGER
├── height: INTEGER
├── created_at: TIMESTAMP
└── processing_status: ENUM('pending', 'annotated', 'processing', 'processed')

annotations
├── id: UUID (PK)
├── image_id: UUID (FK → images)
├── bbox_x: INTEGER (top-left x)
├── bbox_y: INTEGER (top-left y)
├── bbox_width: INTEGER
├── bbox_height: INTEGER
└── created_at: TIMESTAMP

processing_results
├── id: UUID (PK)
├── image_id: UUID (FK → images)
├── mask_blob_path: VARCHAR
├── coco_json_blob_path: VARCHAR
├── processed_at: TIMESTAMP
└── batch_id: UUID (groups images processed together)
```

**Database Initialization:**

Tables are auto-created on backend startup using SQLAlchemy's `create_all()`:

```python
# In main.py lifespan or startup event
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)
    yield
```

This runs before the first request. Existing tables are not modified (use migrations for schema changes in future).

**SAM3 Integration:**

Using `predict_inst` with batched boxes per image:

```python
# Per image processing
inference_state = processor.set_image(pil_image)

# All bboxes for this image (xyxy pixel format)
input_boxes = np.array([[x0, y0, x1, y1], ...])

masks, scores, _ = model.predict_inst(
    inference_state,
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)
# Returns one mask per bbox
```

**API Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/images` | Upload images (multipart/form-data) |
| GET | `/images` | List all images with metadata |
| GET | `/images/{id}` | Get single image metadata |
| DELETE | `/images/{id}` | Delete image and related data |
| POST | `/annotations` | Create bbox annotation |
| GET | `/annotations/{image_id}` | Get annotations for image |
| DELETE | `/annotations/{id}` | Delete annotation |
| POST | `/process` | Start batch processing (list of image IDs) |
| GET | `/process/status` | Get processing progress |
| GET | `/export/{image_id}` | Download COCO JSON for image |

**Configuration:**

Environment variables (`.env`):
```
DATABASE_URL=postgresql://user:pass@localhost:5432/samui
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=...;BlobEndpoint=http://azurite:10000/devstoreaccount1;
AZURE_CONTAINER_NAME=samui-images
SAM3_MODEL_NAME=sam3_hiera_large
API_URL=http://backend:8000
```

**Error Handling:**

- Invalid image format → 400 with message
- Image not found → 404
- Processing already running → 409 Conflict
- SAM3 inference failure → Log error, mark image as failed, continue batch

---

## Implementation Strategy

### Development Approach

**Vertical slices with incremental delivery:**

Build feature-complete slices that can be tested end-to-end:

1. **Phase 1 - Upload Flow**: Infrastructure + upload page + blob storage + DB. Result: Can upload and view images.
2. **Phase 2 - Annotation Flow**: Annotation page + bbox storage. Result: Can draw and persist bboxes.
3. **Phase 3 - Processing Flow**: SAM3 integration + processing page + COCO export. Result: Full working system.

Each phase produces a working increment that can be demonstrated and validated.

### Testing Approach

**Minimal, meaningful tests only:**

- `test_api_images.py`: Upload/list/delete image endpoints
- `test_sam3_inference.py`: SAM3 wrapper produces valid masks from bboxes
- `test_coco_export.py`: COCO JSON structure is valid

No UI tests (Streamlit difficult to test, manual validation sufficient for v1).

### Checkpoint Strategy

Each phase ends with validation before proceeding:

- **Self-review**: Verify implementation matches phase deliverable
- **Code quality**: `uvx ruff check packages/`
- **Code complexity**: `uvx ruff check packages/ --select C901,PLR0912,PLR0915`

These ensure code quality is maintained across AI-assisted development sessions.

---

## Changelog

| Date | Change | Rationale |
|------|--------|-----------|
| 2026-01-08 | Added [P3.0] to convert from uv workspace to isolated packages | Backend requires torch (heavy dependency); isolating packages prevents torch from being installed in frontend's virtual environment |
