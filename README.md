# SAM3 WebUI

A web-based interface for SAM3 image segmentation. Upload images, draw bounding box prompts, run inference, and export results in COCO format.

## Features

- **Two Segmentation Modes**:
  - **Inside Box**: Segment objects within user-drawn bounding boxes
  - **Find All**: Discover all instances matching a text prompt or exemplar boxes
- **Annotation Tools**: Draw bounding boxes with support for positive/negative exemplars
- **Batch Processing**: Process multiple images in a single batch
- **COCO Export**: Download segmentation results in standard COCO format

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- [uv](https://docs.astral.sh/uv/) package manager

### Running with Docker

```bash
# Start all services
docker compose up

# Access the UI
open http://localhost:8501

# API docs
open http://localhost:8000/docs
```

### Local Development

```bash
# Start infrastructure (postgres + azurite)
docker compose up postgres azurite -d

# Copy environment file
cp .env.example .env

# Install and run backend (from packages/samui-backend)
cd packages/samui-backend
uv sync
uv run uvicorn samui_backend.main:app --reload

# Install and run frontend (from packages/samui-frontend, in another terminal)
cd packages/samui-frontend
uv sync
uv run streamlit run src/samui_frontend/app.py
```

### Running Tests

```bash
# Run all tests (from packages/samui-backend)
cd packages/samui-backend
uv run pytest ../../tests/ -v
```

## Project Structure

```
samui/
├── pyproject.toml                 # Dev dependencies (ruff, pytest)
├── docker-compose.yaml            # Local development stack
├── .env.example                   # Environment variables template
├── packages/
│   ├── samui-backend/             # FastAPI + SAM3 (isolated package)
│   │   ├── pyproject.toml
│   │   ├── Dockerfile
│   │   └── src/samui_backend/
│   │       ├── main.py            # FastAPI app entry point
│   │       ├── config.py          # Settings from env vars
│   │       ├── schemas.py         # Pydantic models
│   │       ├── enums.py           # Shared enums (JobStatus, PromptType, etc.)
│   │       ├── db/
│   │       │   ├── database.py    # SQLAlchemy engine
│   │       │   └── models.py      # Image, Annotation, ProcessingJob, ProcessingResult
│   │       ├── routes/
│   │       │   ├── images.py      # Image CRUD endpoints
│   │       │   ├── annotations.py # Annotation CRUD endpoints
│   │       │   ├── processing.py  # SAM3 inference endpoints
│   │       │   └── jobs.py        # Job queue management endpoints
│   │       └── services/
│   │           ├── storage.py     # Azure Blob Storage client
│   │           ├── sam3_inference.py  # SAM3 model wrapper
│   │           ├── job_processor.py   # Background job processing
│   │           └── coco_export.py # COCO JSON generation
│   └── samui-frontend/            # Streamlit (isolated package)
│       ├── pyproject.toml
│       ├── Dockerfile
│       └── src/samui_frontend/
│           ├── app.py             # Streamlit entry point
│           ├── config.py          # API URL setting
│           ├── api.py             # Backend API client
│           ├── models.py          # Frontend data models
│           ├── pages/
│           │   ├── upload.py      # Upload page
│           │   ├── annotation.py  # Annotation page
│           │   ├── processing.py  # Processing page
│           │   ├── jobs.py        # Job queue status page
│           │   └── history.py     # Processing history page
│           └── components/
│               ├── image_gallery.py
│               ├── bbox_annotator.py    # Bounding box drawing
│               ├── mode_toggle.py       # Segmentation mode toggle
│               └── arrow_navigator.py   # Image navigation arrows
└── tests/
    ├── conftest.py                # Test fixtures
    ├── test_api_images.py         # Image API tests
    ├── test_api_annotations.py    # Annotation API tests
    ├── test_api_processing.py     # Processing API tests
    ├── test_api_jobs.py           # Job queue API tests
    ├── test_job_processor.py      # Job processor service tests
    ├── test_sam3_inference.py     # SAM3 service tests
    └── test_coco_export.py        # COCO export tests
```

## API Endpoints

### Images

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/images` | Upload image (multipart/form-data) |
| GET | `/images` | List all images |
| GET | `/images/{id}` | Get image metadata |
| GET | `/images/{id}/data` | Get image binary data |
| PATCH | `/images/{id}` | Update image (e.g., text_prompt for find-all mode) |
| DELETE | `/images/{id}` | Delete image |

### Annotations

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/annotations` | Create annotation with `prompt_type` (segment, positive_exemplar, negative_exemplar) |
| GET | `/annotations/{image_id}` | Get annotations, optionally filtered by `?prompt_type=` |
| DELETE | `/annotations/{id}` | Delete annotation |

### Processing

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/process` | Start batch inference with `mode` (inside_box or find_all) |
| GET | `/process/status` | Get processing progress |
| GET | `/process/mask/{image_id}` | Get mask image, optionally filtered by `?mode=` |
| GET | `/process/export/{image_id}` | Download COCO JSON, optionally filtered by `?mode=` |

### Jobs

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/jobs` | Create a processing job (queued for background execution) |
| GET | `/jobs` | List all jobs (newest first) |
| GET | `/jobs/{job_id}` | Get job details and progress |

### Results

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/results/{result_id}/mask` | Get mask PNG for a specific processing result |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |

## Segmentation Modes

### Inside Box Mode

The default mode. Draw bounding boxes around objects you want to segment. SAM3 will generate precise masks for objects within each box.

**Workflow:**
1. Upload images
2. Select "Inside Box" mode on the annotation page
3. Draw bounding boxes around objects of interest
4. Go to processing page, ensure "Inside Box" mode is selected
5. Click "Process" to run inference
6. Download results in COCO format

### Find All Mode

Discover all instances of objects matching a text description or visual exemplars. Useful for finding multiple similar objects across an image.

**Workflow:**
1. Upload images
2. Select "Find All" mode on the annotation page
3. Enter a text prompt (e.g., "red apples") and/or draw exemplar boxes:
   - **Positive exemplars (+)**: Examples of what to find
   - **Negative exemplars (-)**: Examples of what to exclude
4. Go to processing page, select "Find All" mode
5. Click "Process" to run inference - discovered objects become annotations
6. Download results in COCO format

**Notes:**
- Each mode maintains separate annotations and processing results
- You can process the same image with both modes independently
- Find-all mode creates new annotations from discovered objects (marked as model-generated)

## Configuration

Environment variables (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://samui:samui@localhost:5432/samui` |
| `AZURE_STORAGE_CONNECTION_STRING` | Azure Blob Storage connection | Azurite default |
| `AZURE_CONTAINER_NAME` | Blob container name | `samui-images` |
| `SAM3_MODEL_NAME` | SAM3 model to use | `sam3_hiera_large` |
| `HF_TOKEN` | Hugging Face token for model download | (required) |
| `API_URL` | Backend URL for frontend | `http://localhost:8000` |

## Architecture

- **Frontend (Streamlit)**: Lightweight UI for upload, annotation, and processing
- **Backend (FastAPI)**: REST API handling storage, database, and SAM3 inference
- **PostgreSQL**: Metadata storage (images, annotations, processing results)
- **Azurite/Azure Blob Storage**: Image and mask file storage

The backend auto-creates database tables on startup using SQLAlchemy's `create_all()`.

## Development

### Code Quality

```bash
# Lint
uvx ruff check packages/

# Format
uvx ruff format packages/

# Type check (requires installing types)
uvx mypy packages/
```

### Adding Dependencies

```bash
# Backend (from packages/samui-backend)
cd packages/samui-backend
uv add <package>

# Frontend (from packages/samui-frontend)
cd packages/samui-frontend
uv add <package>
```


## TODO

- Make sure to use dataclasses instead of dicts where appropriate
- GPU hardware detection and conditionals for CUDA/CPU
- Replace rectangle drawing component to be able to see live-drawing of bboxes
- Add SAM 1 style point prompts
