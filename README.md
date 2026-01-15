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
# Start all services and build images
docker compose up --build

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

## Segmentation Modes

### Inside Box Mode

The default mode. Draw bounding boxes around objects you want to segment. SAM3 will generate masks for objects within each box.

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
5. Click "Process" to run inference
6. Download results in COCO format

**Notes:**
- Each mode maintains separate annotations and processing results
- You can process the same image with both modes independently

## Architecture

Monorepo with isolated packages: `packages/samui-backend` (FastAPI) and `packages/samui-frontend` (Streamlit).

```
┌─────────────────────────────────────────────────────────────────┐
│                           Frontend                              │
│                       (Streamlit :8501)                         │
│  ┌────────┐ ┌──────────┐ ┌──────────┐ ┌──────┐ ┌─────────┐      │
│  │ Upload │ │Annotation│ │Processing│ │ Jobs │ │ History │      │
│  └────────┘ └──────────┘ └──────────┘ └──────┘ └─────────┘      │
└────────────────────────────────┬────────────────────────────────┘
                                 │ HTTP/REST
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                            Backend                              │
│                        (FastAPI :8000)                          │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                       Routes Layer                        │  │
│  │  /images   /annotations   /processing   /jobs   /results  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                  │
│  ┌───────────────────────────┴───────────────────────────────┐  │
│  │                     Services Layer                        │  │
│  │  ┌───────────┐  ┌─────────────┐  ┌───────────────────┐    │  │
│  │  │  Storage  │  │    SAM3     │  │   JobProcessor    │    │  │
│  │  │  Service  │  │  Inference  │  │ (background task) │    │  │
│  │  └─────┬─────┘  └──────┬──────┘  └─────────┬─────────┘    │  │
│  │        │               │                   │              │  │
│  │        │        ┌──────┴──────┐            │              │  │
│  │        │        │ SAM3 Model  │            │              │  │
│  │        │        │    (GPU)    │            │              │  │
│  │        │        └─────────────┘            │              │  │
│  └────────┼───────────────────────────────────┼──────────────┘  │
│           │                                   │                 │
└───────────┼───────────────────────────────────┼─────────────────┘
            │                                   │
            ▼                                   ▼
┌───────────────────────┐           ┌───────────────────────┐
│  Azure Blob Storage   │           │      PostgreSQL       │
│      (Azurite)        │           │                       │
│                       │           │ ┌───────────────────┐ │
│  • Original images    │           │ │      images       │ │
│  • Mask PNGs          │           │ ├───────────────────┤ │
│  • COCO JSON exports  │           │ │   annotations     │ │
│                       │           │ ├───────────────────┤ │
└───────────────────────┘           │ │  processing_jobs  │ │
                                    │ ├───────────────────┤ │
                                    │ │processing_results │ │
                                    │ └───────────────────┘ │
                                    └───────────────────────┘
```

### Component Responsibilities

| Component | Purpose |
|-----------|---------|
| **Frontend** | Streamlit UI for image upload, bounding box annotation, triggering processing, and viewing results |
| **Backend** | FastAPI REST API that orchestrates all operations and hosts the SAM3 model |
| **Storage Service** | Abstracts blob storage operations (upload/download images and masks) |
| **SAM3 Inference** | Wraps SAM3 model with lazy loading - loads on first inference, unloads to free GPU memory |
| **Job Processor** | Runs inference jobs in background tasks with queue support (one job runs at a time) |
| **PostgreSQL** | Stores metadata: image records, annotations, job status, and processing results |
| **Blob Storage** | Stores binary data: original images, generated masks, and COCO JSON exports |

### Data Flow

1. **Upload**: Frontend → `/images` → Storage Service saves to blob → DB stores metadata
2. **Annotate**: Frontend → `/annotations` → DB stores bounding boxes
3. **Process**: Frontend → `/jobs` → JobProcessor queues work → SAM3 runs inference → masks saved to blob → results saved to DB
4. **Export**: Frontend → `/results/{id}/mask` or `/process/export/{id}` → retrieves from blob storage

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

## Development

### Code Quality

```bash
# Lint
uvx ruff check packages/

# Format
uvx ruff format packages/
```

### Adding Dependencies

```bash
# Backend
cd packages/samui-backend && uv add <package>

# Frontend
cd packages/samui-frontend && uv add <package>
```
