# SAM3 WebUI Tasklist

> **UI Reference:** See `plans/sam3ui-mockup.html` for interactive mockup of all pages.

## Phase 1: Upload Flow

**Goal:** Establish project infrastructure and implement image upload with storage and viewing.

**Deliverable:** Working upload page where users can upload images, store them in blob storage, persist metadata in PostgreSQL, and view uploaded images in a tiled gallery.

**Tasks:**

- [ ] [P1.1] Create root `pyproject.toml` as uv workspace with members `packages/samui-backend` and `packages/samui-frontend`, dev dependencies (pytest, ruff)
- [ ] [P1.2] Create `packages/samui-backend/pyproject.toml` with dependencies (fastapi, uvicorn, sqlalchemy, psycopg2-binary, azure-storage-blob, pydantic, pillow, numpy) and src layout
- [ ] [P1.3] Create `packages/samui-frontend/pyproject.toml` with dependencies (streamlit, streamlit-image-coordinates, httpx, pydantic, pillow) and src layout
- [ ] [P1.4] Create `docker-compose.yaml` with services: postgres, azurite, backend (builds samui-backend), frontend (builds samui-frontend)
- [ ] [P1.5] Create `packages/samui-backend/Dockerfile` for FastAPI service
- [ ] [P1.6] Create `packages/samui-frontend/Dockerfile` for Streamlit service
- [ ] [P1.7] Create `.env.example` with DATABASE_URL, AZURE_STORAGE_CONNECTION_STRING, AZURE_CONTAINER_NAME, SAM3_MODEL_NAME, API_URL
- [ ] [P1.8] Create `packages/samui-backend/src/samui_backend/__init__.py` and `config.py` with pydantic Settings class loading env vars
- [ ] [P1.9] Create `packages/samui-backend/src/samui_backend/db/database.py` with SQLAlchemy engine, SessionLocal, and Base
- [ ] [P1.10] Create `packages/samui-backend/src/samui_backend/db/models.py` with Image model (id, filename, blob_path, width, height, created_at, processing_status enum)
- [ ] [P1.11] Create `packages/samui-backend/src/samui_backend/services/storage.py` with StorageService class (upload_image, get_image, delete_image, get_image_url methods)
- [ ] [P1.12] Create `packages/samui-backend/src/samui_backend/schemas.py` with Pydantic models: ImageCreate, ImageResponse, ImageList
- [ ] [P1.13] Create `packages/samui-backend/src/samui_backend/routes/images.py` with POST /images (upload), GET /images (list), GET /images/{id}, DELETE /images/{id}
- [ ] [P1.14] Create `packages/samui-backend/src/samui_backend/main.py` with FastAPI app, lifespan that calls `create_all()` for DB init, include images router, add CORS middleware
- [ ] [P1.15] Create `packages/samui-frontend/src/samui_frontend/__init__.py` and `config.py` with API_URL setting
- [ ] [P1.16] Create `packages/samui-frontend/src/samui_frontend/components/image_gallery.py` with reusable tiled image gallery component
- [ ] [P1.17] Create `packages/samui-frontend/src/samui_frontend/pages/upload.py` with drag-and-drop upload, file selector, and image gallery display
- [ ] [P1.18] Create `packages/samui-frontend/src/samui_frontend/app.py` as Streamlit entry point with page navigation
- [ ] [P1.19] Create `tests/conftest.py` with test database fixture and test client
- [ ] [P1.20] Create `tests/test_api_images.py` with tests for upload, list, and delete endpoints
- [ ] [P1.21] Run tests: `uv run pytest tests/test_api_images.py -v`
- [ ] [P1.22] Verify full flow manually: `docker compose up`, upload images via UI, confirm images appear in gallery

**Checkpoints:**

- [ ] Code quality: Run `uvx ruff check packages/`
- [ ] Code complexity: Run `uvx ruff check packages/ --select C901,PLR0912,PLR0915`
- [ ] Review: Verify upload page works end-to-end (upload → storage → DB → gallery display)

**Phase 1 Complete:** Infrastructure established with split packages. Users can upload images via drag-and-drop, images are stored in Azurite blob storage with metadata in PostgreSQL (auto-initialized on backend startup), and uploaded images display in a tiled gallery. Frontend container is lightweight (no torch).

---

## Phase 2: Annotation Flow

**Goal:** Implement bounding box annotation on images with persistence.

**Deliverable:** Working annotation page where users can view a large image, draw multiple bounding boxes (displayed in different colors), see annotation details in a side panel, navigate between images, and have annotations persisted to the database.

**Tasks:**

- [ ] [P2.1] Add Annotation model to `packages/samui-backend/src/samui_backend/db/models.py` (id, image_id FK, bbox_x, bbox_y, bbox_width, bbox_height, created_at)
- [ ] [P2.2] Add annotation schemas to `packages/samui-backend/src/samui_backend/schemas.py`: AnnotationCreate, AnnotationResponse, AnnotationList
- [ ] [P2.3] Create `packages/samui-backend/src/samui_backend/routes/annotations.py` with POST /annotations, GET /annotations/{image_id}, DELETE /annotations/{id}
- [ ] [P2.4] Register annotations router in `packages/samui-backend/src/samui_backend/main.py`
- [ ] [P2.5] Prototype streamlit-image-coordinates: verify rectangle selection and multi-bbox overlay rendering meets requirements before building full annotator
- [ ] [P2.6] Create `packages/samui-frontend/src/samui_frontend/components/bbox_annotator.py` using streamlit-image-coordinates for rectangle selection
  - Display image at large size
  - Capture rectangle coordinates on draw
  - Render existing bboxes as colored overlays (different color per bbox)
  - Return bbox coordinates to parent
- [ ] [P2.7] Create `packages/samui-frontend/src/samui_frontend/pages/annotation.py` with page layout and bbox_annotator component integration
- [ ] [P2.8] Add annotation list panel to annotation.py with delete buttons and API calls to create/delete annotations
- [ ] [P2.9] Add image navigation to annotation.py: tiled gallery for selection and arrow key navigation between images
- [ ] [P2.10] Update `packages/samui-frontend/src/samui_frontend/app.py` to include annotation page in navigation
- [ ] [P2.11] Update Image model processing_status to 'annotated' when first annotation added (in annotations route)
- [ ] [P2.12] Create `tests/test_api_annotations.py` with tests for annotation endpoints:
  - POST /annotations: success, image not found (404), invalid bbox (400), multiple annotations same image
  - GET /annotations/{image_id}: returns list, empty list, image not found (404)
  - DELETE /annotations/{id}: success, not found (404)
  - Status update: first annotation sets image status to 'annotated'
- [ ] [P2.13] Run tests: `uv run pytest tests/test_api_annotations.py -v`
- [ ] [P2.14] Verify annotation flow manually: navigate to image, draw bboxes, see them persist on page refresh, delete annotations

**Checkpoints:**

- [ ] Code quality: Run `uvx ruff check packages/`
- [ ] Code complexity: Run `uvx ruff check packages/ --select C901,PLR0912,PLR0915`
- [ ] Review: Verify annotation page works (draw bbox → save → reload → bbox visible, delete works, navigation works)

**Phase 2 Complete:** Annotation system working. Users can select images, draw multiple bounding boxes with visual feedback, view/delete annotations, and navigate between images. Annotations persist to database.

---

## Phase 3: Processing Flow

**Goal:** Implement SAM3 inference, processing status tracking, and COCO export.

**Deliverable:** Working processing page where users can trigger batch SAM3 inference on annotated images, see real-time progress, view processed images with mask overlays, and download COCO JSON annotations per image.

**Tasks:**

- [ ] [P3.1] Add `sam3` and `torch` dependencies to `packages/samui-backend/pyproject.toml`
- [ ] [P3.2] Add ProcessingResult model to `packages/samui-backend/src/samui_backend/db/models.py` (id, image_id FK, mask_blob_path, coco_json_blob_path, processed_at, batch_id)
- [ ] [P3.3] Create `packages/samui-backend/src/samui_backend/services/sam3_inference.py` with SAM3Service class:
  - `load_model()`: Load SAM3 with `build_sam3_image_model` and create processor
  - `process_image(image, bboxes) -> masks`: Use `predict_inst` with batched boxes (xyxy format)
  - `unload_model()`: Release model from memory
- [ ] [P3.4] Create `packages/samui-backend/src/samui_backend/services/coco_export.py` with function to generate COCO JSON from image metadata, bboxes, and masks
  - One JSON per image
  - Include image info, annotations array with segmentation (RLE), bbox, area, category_id
- [ ] [P3.5] Add processing schemas to `packages/samui-backend/src/samui_backend/schemas.py`: ProcessRequest (list of image_ids), ProcessStatus, ExportResponse
- [ ] [P3.6] Create `packages/samui-backend/src/samui_backend/routes/processing.py` with:
  - POST /process: Accept image_ids, start BackgroundTask, return batch_id
  - GET /process/status: Return current progress (processed count, total, current image)
  - Background task logic: load model, iterate images, fetch from storage, run inference, save masks, generate COCO JSON, update DB status per image, unload model
  - Implement idempotent processing: skip images where processing_status == 'processed' and mask_blob_path exists (allows safe restart after crash)
- [ ] [P3.7] Add GET /export/{image_id} endpoint to processing routes: fetch COCO JSON from blob storage
- [ ] [P3.8] Register processing router in `packages/samui-backend/src/samui_backend/main.py`
- [ ] [P3.9] Create `packages/samui-frontend/src/samui_frontend/pages/processing.py` with process button (calls POST /process with annotated image IDs) and progress indicator polling GET /process/status
- [ ] [P3.10] Add processed image viewer to processing.py: large image view with mask overlay and tiled gallery of processed images
- [ ] [P3.11] Add export and navigation to processing.py: download button per image calling GET /export/{image_id} and arrow key navigation
- [ ] [P3.12] Update `packages/samui-frontend/src/samui_frontend/app.py` to include processing page in navigation
- [ ] [P3.13] Create `tests/test_sam3_inference.py` testing SAM3Service.process_image with mock image and bboxes
- [ ] [P3.14] Create `tests/test_coco_export.py` testing COCO JSON generation produces valid structure
- [ ] [P3.15] Run tests: `uv run pytest tests/ -v`
- [ ] [P3.16] Verify full flow manually: upload images → annotate with bboxes → process → view masks → download COCO JSON

**Checkpoints:**

- [ ] Code quality: Run `uvx ruff check packages/`
- [ ] Code complexity: Run `uvx ruff check packages/ --select C901,PLR0912,PLR0915`
- [ ] Review: Verify complete flow works end-to-end (upload → annotate → process → export)

**Phase 3 Complete:** Full SAM3 WebUI operational. Users can upload images, annotate with bounding boxes, process through SAM3 with progress tracking, view segmentation results, and export COCO-format annotations.
