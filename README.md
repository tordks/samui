# SAM3 WebUI

A web-based interface for SAM3 image segmentation. Upload images, draw bounding box prompts, run inference, and export results in COCO format.

## Current Status

**Phase 1: Upload Flow** - Complete

- Image upload via drag-and-drop
- Azure Blob Storage integration (Azurite for local dev)
- PostgreSQL metadata persistence
- Tiled image gallery display

**Phase 2: Annotation Flow** - Complete

- Bounding box annotation with click-and-drag
- Multi-bbox support with colored overlays
- Annotation list sidebar with delete buttons
- Image navigation (previous/next, thumbnail gallery)
- Auto-status update to 'annotated' on first annotation

**Phase 3: Processing Flow** - Complete

- SAM3 inference with batched box prompts (predict_inst API)
- Background task processing with progress tracking
- COCO JSON export with RLE segmentation format
- Processed image viewer with mask display
- Per-image COCO download

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
│   │       ├── db/
│   │       │   ├── database.py    # SQLAlchemy engine
│   │       │   └── models.py      # Image, Annotation, ProcessingResult
│   │       ├── routes/
│   │       │   ├── images.py      # Image CRUD endpoints
│   │       │   ├── annotations.py # Annotation CRUD endpoints
│   │       │   └── processing.py  # SAM3 inference endpoints
│   │       └── services/
│   │           ├── storage.py     # Azure Blob Storage client
│   │           ├── sam3_inference.py  # SAM3 model wrapper
│   │           └── coco_export.py # COCO JSON generation
│   └── samui-frontend/            # Streamlit (isolated package)
│       ├── pyproject.toml
│       ├── Dockerfile
│       └── src/samui_frontend/
│           ├── app.py             # Streamlit entry point
│           ├── config.py          # API URL setting
│           ├── pages/
│           │   ├── upload.py      # Upload page
│           │   ├── annotation.py  # Annotation page
│           │   └── processing.py  # Processing page
│           └── components/
│               ├── image_gallery.py
│               └── bbox_annotator.py  # Bounding box drawing
└── tests/
    ├── conftest.py                # Test fixtures
    ├── test_api_images.py         # Image API tests
    ├── test_api_annotations.py    # Annotation API tests
    ├── test_sam3_inference.py     # SAM3 service tests
    └── test_coco_export.py        # COCO export tests
```

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

## API Endpoints

### Images

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/images` | Upload image (multipart/form-data) |
| GET | `/images` | List all images |
| GET | `/images/{id}` | Get image metadata |
| GET | `/images/{id}/data` | Get image binary data |
| DELETE | `/images/{id}` | Delete image |

### Annotations

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/annotations` | Create bounding box annotation |
| GET | `/annotations/{image_id}` | Get all annotations for an image |
| DELETE | `/annotations/{id}` | Delete annotation |

### Processing

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/process` | Start batch SAM3 inference |
| GET | `/process/status` | Get processing progress |
| GET | `/process/export/{image_id}` | Download COCO JSON for image |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |

## Configuration

Environment variables (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://samui:samui@localhost:5432/samui` |
| `AZURE_STORAGE_CONNECTION_STRING` | Azure Blob Storage connection | Azurite default |
| `AZURE_CONTAINER_NAME` | Blob container name | `samui-images` |
| `SAM3_MODEL_NAME` | SAM3 model to use | `sam3_hiera_large` |
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

- Use pytorch GPU enabled image for backend Docker container
- Make os.path -> pathlib throughout codebase
- Make sure to use dataclasses instead of dicts where appropriate
- GPU hardware detection and conditionals for CUDA/MPS/CPU