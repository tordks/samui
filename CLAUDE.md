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

# Backend
cd packages/samui-backend
uv sync
uv run uvicorn samui_backend.main:app --reload

# Frontend (in another terminal)
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

Tests use in-memory SQLite and mocked Azure storage. Fixtures are in `tests/conftest.py`.

### Linting and Formatting

```bash
# Lint
uvx ruff check packages/

# Format
uvx ruff format packages/

# Type check
uvx mypy packages/
```

### Adding Dependencies

```bash
# Backend
cd packages/samui-backend && uv add <package>

# Frontend
cd packages/samui-frontend && uv add <package>
```

## Architecture

Monorepo with isolated packages:

- `packages/samui-backend/` - FastAPI REST API with SAM3 inference
  - `routes/` - API endpoints (images, annotations, processing)
  - `services/` - Business logic (storage, sam3_inference, coco_export)
  - `db/` - SQLAlchemy models and database setup
  - `schemas.py` - Pydantic request/response models

- `packages/samui-frontend/` - Streamlit UI
  - `pages/` - Upload, annotation, processing pages
  - `components/` - Reusable UI components

- `tests/` - Integration tests at project root (run from backend package)

**Key patterns:**
- FastAPI dependency injection for services and database sessions
- SQLAlchemy ORM with UUID primary keys and type-mapped columns
- Pydantic for all API validation
- Database tables auto-created on startup via lifespan context manager
