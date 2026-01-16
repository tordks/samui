# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAM3 WebUI - A full-stack web application for SAM3 (Segment Anything Model 3) image segmentation. Users upload images, draw bounding box prompts, run inference, and export results in COCO format.

**Tech stack:** FastAPI backend, Streamlit frontend, PostgreSQL database, Azure Blob Storage (Azurite for local dev), PyTorch 2.7 with CUDA support.

## Development Commands

### Running Services

```bash
# Docker: Start full stack
docker compose up

# Docker: Infrastructure only (for local Python development)
docker compose up postgres azurite -d

# Copy environment file (first time setup)
cp .env.example .env

# Backend (API at http://localhost:8000, docs at http://localhost:8000/docs)
cd packages/samui-backend
uv sync
uv run uvicorn samui_backend.main:app --reload

# Frontend (UI at http://localhost:8501)
cd packages/samui-frontend
uv sync
uv run streamlit run src/samui_frontend/app.py
```

### Testing

```bash
# Run all tests (from packages/samui-backend)
cd packages/samui-backend
uv run pytest ../../tests/ -v

# Run single test file
uv run pytest ../../tests/test_api_images.py -v

# Run specific test
uv run pytest ../../tests/test_api_images.py::test_upload_image -v
```

Tests use in-memory SQLite and mocked Azure storage. Must run from `packages/samui-backend` to resolve dependencies. Fixtures in `tests/conftest.py` use FastAPI's `dependency_overrides` to inject test database and mock storage.

### Linting and Formatting

```bash
# Lint
uvx ruff check packages/

# Format
uvx ruff format packages/

# Security scan (uses config from pyproject.toml)
uvx bandit -c pyproject.toml -r .
```

### Dead Code Detection

```bash
# Find unused code in backend
uvx vulture packages/samui-backend/src/

# Find unused code in frontend
uvx vulture packages/samui-frontend/src/

# Check both packages
uvx vulture packages/samui-backend/src/ packages/samui-frontend/src/

# Higher confidence threshold (default 60%)
uvx vulture packages/samui-backend/src/ --min-confidence 80
```

**Finding code only used in tests:** Run vulture on source only, then on source + tests. Code appearing in the first output but not the second is only referenced by tests.

```bash
# Source only - shows all potentially unused code
uvx vulture packages/samui-backend/src/ packages/samui-frontend/src/

# Source + tests - unused code here is truly dead
uvx vulture packages/samui-backend/src/ packages/samui-frontend/src/ tests/
```

**Note:** Vulture may report false positives for Pydantic models, SQLAlchemy columns, and FastAPI dependencies since they're used implicitly by frameworks.

### Adding Dependencies

```bash
# Backend
cd packages/samui-backend && uv add <package>

# Frontend
cd packages/samui-frontend && uv add <package>
```

### Project Tree with Line Counts

```bash
# From current directory
tree -fi --noreport | while read f; do [ -f "$f" ] && printf "%s (%d lines)\n" "$f" "$(wc -l < "$f")" || echo "$f"; done

# From custom root
tree -fi --noreport packages/samui-backend/src | while read f; do [ -f "$f" ] && printf "%s (%d lines)\n" "$f" "$(wc -l < "$f")" || echo "$f"; done
```

Files should be under 500 lines for maintainability. Consider refactoring large files into smaller modules if writing to a large file.

## Architecture

Monorepo with isolated packages:

- `packages/samui-backend/` - FastAPI REST API with SAM3 inference
  - `routes/` - API endpoints (images, annotations, processing)
  - `services/` - Business logic (storage, sam3_inference, coco_export)
  - `db/` - SQLAlchemy models and database setup
  - `schemas.py` - Pydantic request/response models

- `packages/samui-frontend/` - Streamlit UI
  - `pages/` - Upload, annotation (bbox and point), processing pages
  - `components/` - Reusable UI components (annotators, gallery, controls)

- `tests/` - Integration tests at project root (run from backend package)

**Key patterns:**
- FastAPI dependency injection for services and database sessions
- SQLAlchemy ORM with UUID primary keys and type-mapped columns
- Pydantic for all API validation
- Database tables auto-created on startup via lifespan context manager
- SAM3 model uses lazy loading (`load_model()` before inference, `unload_model()` to free GPU)

## Segmentation Modes

The application supports three segmentation modes, each with distinct annotation types and processing methods:

### Inside Box Mode (default)
- User draws bounding boxes (prompt_type=SEGMENT)
- SAM3 segments objects within each box using `process_image()` with interactive API
- One mask per annotation

### Find All Mode
- User provides text prompt (stored on Image.text_prompt) and/or exemplar boxes
- Annotations use prompt_type=POSITIVE_EXEMPLAR or NEGATIVE_EXEMPLAR
- SAM3 discovers all matching instances using `process_image_find_all()` with batched API
- Creates new annotations (source=MODEL) for discovered objects

### Point Mode
- User clicks on images to place positive (foreground) and negative (background) points
- Uses `PointAnnotation` model with x, y coordinates and is_positive flag
- SAM3 inference using `predict_inst()` with point coordinates/labels
- One mask per image from combined point prompts

**Key data model fields:**
- `BboxAnnotation.prompt_type` - SEGMENT, POSITIVE_EXEMPLAR, or NEGATIVE_EXEMPLAR
- `BboxAnnotation.source` - USER or MODEL (model-generated from find-all)
- `PointAnnotation` - x, y, is_positive for point mode
- `Image.text_prompt` - Text description for find-all mode
- `ProcessingResult.mode` - INSIDE_BOX, FIND_ALL, or POINT (unique per image+mode)

**Frontend session state:**
- Both annotation and processing pages share `segmentation_mode` in session state
- Mode affects which annotations are fetched, displayed, and processed

## Environment Variables

Key variables in `.env.example`:
- `DATABASE_URL` - PostgreSQL connection string
- `AZURE_STORAGE_CONNECTION_STRING` - Blob storage (Azurite for local dev)
- `HF_TOKEN` - Required for SAM3 model download from Hugging Face
