# SAM3 WebUI

A web-based interface for SAM3 image segmentation. Upload images, draw bounding box prompts, run inference, and export results in COCO format.

## Current Status

**Phase 1: Upload Flow** - Complete

- Image upload via drag-and-drop
- Azure Blob Storage integration (Azurite for local dev)
- PostgreSQL metadata persistence
- Tiled image gallery display

**Phase 2: Annotation Flow** - Not started

**Phase 3: Processing Flow** - Not started

## Project Structure

```
samui/
├── pyproject.toml                 # uv workspace root
├── docker-compose.yaml            # Local development stack
├── .env.example                   # Environment variables template
├── packages/
│   ├── samui-backend/             # FastAPI + SAM3 (heavy deps)
│   │   ├── pyproject.toml
│   │   ├── Dockerfile
│   │   └── src/samui_backend/
│   │       ├── main.py            # FastAPI app entry point
│   │       ├── config.py          # Settings from env vars
│   │       ├── schemas.py         # Pydantic models
│   │       ├── db/
│   │       │   ├── database.py    # SQLAlchemy engine
│   │       │   └── models.py      # Image model
│   │       ├── routes/
│   │       │   └── images.py      # Image CRUD endpoints
│   │       └── services/
│   │           └── storage.py     # Azure Blob Storage client
│   └── samui-frontend/            # Streamlit (light deps)
│       ├── pyproject.toml
│       ├── Dockerfile
│       └── src/samui_frontend/
│           ├── app.py             # Streamlit entry point
│           ├── config.py          # API URL setting
│           ├── pages/
│           │   └── upload.py      # Upload page
│           └── components/
│               └── image_gallery.py
└── tests/
    ├── conftest.py                # Test fixtures
    └── test_api_images.py         # Image API tests
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
# Install dependencies
uv sync

# Install backend package
uv sync --package samui-backend

# Start infrastructure (postgres + azurite)
docker compose up postgres azurite -d

# Copy environment file
cp .env.example .env

# Run backend
uv run --package samui-backend uvicorn samui_backend.main:app --reload

# Run frontend (in another terminal)
uv run --package samui-frontend streamlit run packages/samui-frontend/src/samui_frontend/app.py
```

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ -v --cov=packages
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/images` | Upload image (multipart/form-data) |
| GET | `/images` | List all images |
| GET | `/images/{id}` | Get image metadata |
| GET | `/images/{id}/data` | Get image binary data |
| DELETE | `/images/{id}` | Delete image |
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
# Backend
uv add --package samui-backend <package>

# Frontend
uv add --package samui-frontend <package>

# Dev dependencies (workspace root)
uv add --dev <package>
```
