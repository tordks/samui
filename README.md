# SAM3 WebUI

A web-based interface for SAM3 image segmentation. Upload images, provide prompts (bounding boxes or points), run inference, and export results in COCO format.

## Features

- **Three Segmentation Modes**:
  - **Inside Box**: Segment objects within user-drawn bounding boxes
  - **Find All**: Discover all instances matching a text prompt or exemplar boxes
  - **Point**: Click to place positive/negative points for precise segmentation control
- **Annotation Tools**: Draw bounding boxes or place points with support for positive/negative prompts
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
2. Select "Inside Box" mode on the Annotation page
3. Draw bounding boxes around objects of interest
4. Click "Process" to run inference
5. View results on History page and download in COCO format

### Find All Mode

Discover all instances of objects matching a text description or visual exemplars. Useful for finding multiple similar objects across an image.

**Workflow:**
1. Upload images
2. Select "Find All" mode on the Annotation page
3. Enter a text prompt (e.g., "red apples") and/or draw exemplar boxes:
   - **Positive exemplars (+)**: Examples of what to find
   - **Negative exemplars (-)**: Examples of what to exclude
4. Click "Process" to run inference
5. View results on History page and download in COCO format

### Point Mode

Click directly on objects to indicate what to segment. Offers precise control for interactive segmentation.

**Workflow:**
1. Upload images
2. Go to the Point Annotation page
3. Click on the image to place points:
   - **Positive points (green)**: Indicate foreground (what to segment)
   - **Negative points (red)**: Indicate background (what to exclude)
4. Click "Process" to run inference
5. View results on History page and download in COCO format

**Notes:**
- Each mode maintains separate annotations and processing results
- You can process the same image with all three modes independently

## Architecture

Monorepo with isolated packages: `packages/samui-backend` (FastAPI) and `packages/samui-frontend` (Streamlit).

```
┌─────────────────────────────────────────────────────────────────┐
│                           Frontend                              │
│                       (Streamlit :8501)                         │
│  ┌────────┐ ┌──────────┐ ┌───────┐ ┌─────────┐ ┌──────┐         │
│  │ Upload │ │Annotation│ │ Point │ │ History │ │ Jobs │         │
│  └────────┘ └──────────┘ └───────┘ └─────────┘ └──────┘         │
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
│  • COCO JSON exports  │           │ │ bbox_annotations  │ │
│                       │           │ ├───────────────────┤ │
└───────────────────────┘           │ │ point_annotations │ │
                                    │ ├───────────────────┤ │
                                    │ │  processing_jobs  │ │
                                    │ ├───────────────────┤ │
                                    │ │processing_results │ │
                                    │ └───────────────────┘ │
                                    └───────────────────────┘
```

### Component Responsibilities

| Component | Purpose |
|-----------|---------|
| **Frontend** | Streamlit UI for image upload, annotation (bounding boxes and points), triggering processing, and viewing results |
| **Backend** | FastAPI REST API that orchestrates all operations and hosts the SAM3 model |
| **Storage Service** | Abstracts blob storage operations (upload/download images and masks) |
| **SAM3 Inference** | Wraps SAM3 model with lazy loading - loads on first inference, unloads to free GPU memory |
| **Job Processor** | Runs inference jobs in background tasks with queue support (one job runs at a time) |
| **PostgreSQL** | Stores metadata: image records, bbox/point annotations, job status, and processing results |
| **Blob Storage** | Stores binary data: original images, generated masks, and COCO JSON exports |

### Data Flow

1. **Upload**: Frontend → `/images` → Storage Service saves to blob → DB stores metadata
2. **Annotate**: Frontend → `/annotations` → DB stores bounding boxes or points
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
