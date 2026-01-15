# Point Mode Tasklist

Step-by-step implementation guide for point-based segmentation mode.

---

## Phase 1: Data Model Refactor

**Goal:** Split Annotation table into BboxAnnotation and PointAnnotation

**Deliverable:** Clean data model with separate tables for bbox and point annotations, all existing tests passing

**Tasks:**

- [x] [P1.1] Rename `Annotation` class to `BboxAnnotation` in `packages/samui-backend/src/samui_backend/db/models.py`
  - Update class name and `__tablename__` to `bbox_annotations`
  - Keep all existing fields: id, image_id, bbox_x, bbox_y, bbox_width, bbox_height, prompt_type, source, created_at

- [x] [P1.2] Create `PointAnnotation` model in `packages/samui-backend/src/samui_backend/db/models.py`
  - Fields: id (UUID), image_id (FK), point_x (Integer), point_y (Integer), is_positive (Boolean), created_at (DateTime)
  - Add relationship to Image model
  - Add `POINT` to `SegmentationMode` enum in `enums.py`

- [x] [P1.3] Update `packages/samui-backend/src/samui_backend/schemas.py` for model changes
  - Rename `AnnotationCreate`, `AnnotationResponse` to `BboxAnnotationCreate`, `BboxAnnotationResponse`
  - Create `PointAnnotationCreate` schema: image_id, point_x, point_y, is_positive
  - Create `PointAnnotationResponse` schema: id, image_id, point_x, point_y, is_positive, created_at

- [x] [P1.4] Update `packages/samui-backend/src/samui_backend/routes/annotations.py` for BboxAnnotation
  - Update imports: `Annotation` → `BboxAnnotation`
  - Update schema references: `AnnotationCreate` → `BboxAnnotationCreate`, etc.
  - Update all database queries to use `BboxAnnotation`

- [x] [P1.5] Update `packages/samui-backend/src/samui_backend/services/job_processor.py` for refactored models
  - Update imports: `Annotation` → `BboxAnnotation`
  - Update `get_annotations_for_mode()` to query `BboxAnnotation`

- [x] [P1.6] Update `packages/samui-backend/src/samui_backend/services/coco_export.py` for BboxAnnotation
  - Update imports and type hints: `Annotation` → `BboxAnnotation`

- [x] [P1.7] Update `packages/samui-frontend/src/samui_frontend/api.py` for BboxAnnotation
  - Update annotation function names if endpoints change

- [x] [P1.8] Update `packages/samui-frontend/src/samui_frontend/pages/annotation.py` for BboxAnnotation
  - Update any references to annotation schema names if changed

- [x] [P1.9] Update `tests/test_api_annotations.py` for BboxAnnotation
  - Update imports and references: `Annotation` → `BboxAnnotation`
  - Ensure all existing annotation tests pass with renamed model

- [x] [P1.10] Update `tests/test_job_processor.py` for refactored models
  - Update imports: `Annotation` → `BboxAnnotation`
  - Update test fixtures and assertions

- [x] [P1.11] Run full test suite to verify no regressions: `cd packages/samui-backend && uv run pytest ../../tests/ -v`

**Checkpoints:**

- [x] Code quality: Run `uvx ruff check packages/ --fix`
- [x] Code formatting: Run `uvx ruff format packages/`
- [x] Review: Verify all existing modes (INSIDE_BOX, FIND_ALL) work correctly after refactor

**Phase 1 Complete:** Data model cleanly separated into BboxAnnotation and PointAnnotation tables. All existing tests pass, ready for point mode implementation.

---

## Phase 2: Backend Point Support

**Goal:** Add PointAnnotation CRUD API and SAM3 point-based inference

**Deliverable:** Working backend API for point annotations and point-based segmentation processing

**Tasks:**

- [x] [P2.1] Create point annotation routes in `packages/samui-backend/src/samui_backend/routes/annotations.py`
  - Add `POST /point-annotations` endpoint to create point annotation
  - Add `GET /point-annotations/{image_id}` endpoint to list points for image
  - Add `DELETE /point-annotations/{annotation_id}` endpoint to delete point
  - Include validation: coordinates within image bounds, is_positive boolean

- [x] [P2.2] Add `process_image_points()` method to `packages/samui-backend/src/samui_backend/services/sam3_inference.py`
  - Accept image, points list (x, y tuples), labels list (1=positive, 0=negative)
  - Convert to numpy arrays with correct shapes
  - Call `predict_inst()` with point_coords and point_labels
  - Return single combined mask

- [x] [P2.3] Refactor `process_single_image()` into mode-specific helpers in `job_processor.py`
  - Add `get_point_annotations_for_image()` helper function
  - Update `needs_processing()` to check PointAnnotation for POINT mode
  - Extract `_process_inside_box()` helper from existing INSIDE_BOX logic
  - Extract `_process_find_all()` helper from existing FIND_ALL logic
  - Create `_process_point()` helper for POINT mode:
    - Fetch point annotations
    - Extract coordinates and labels
    - Call `sam3.process_image_points()`
    - Save mask and COCO JSON
  - Refactor `process_single_image()` to dispatch to helpers

- [x] [P2.4] Update `packages/samui-backend/src/samui_backend/services/coco_export.py` to support point annotations
  - Add function to convert point annotations to COCO format
  - Include point coordinates in annotation metadata

- [x] [P2.5] Create `tests/test_api_point_annotations.py` with point annotation API tests
  - Test create point annotation (valid coordinates)
  - Test create point annotation (invalid: out of bounds)
  - Test list points for image
  - Test delete point annotation
  - Test points filtering by image_id

- [x] [P2.6] Add point inference tests to `tests/test_sam3_inference.py`
  - Test `process_image_points()` with mocked model
  - Test with positive points only
  - Test with mixed positive/negative points
  - Verify correct numpy array shapes passed to model

- [x] [P2.7] Add POINT mode tests to `tests/test_job_processor.py`
  - Test `needs_processing()` for POINT mode
  - Test `get_point_annotations_for_image()` returns PointAnnotations
  - Test `_process_point()` helper function

- [x] [P2.8] Run test suite: `cd packages/samui-backend && uv run pytest ../../tests/ -v`

**Checkpoints:**

- [x] Code quality: Run `uvx ruff check packages/ --fix`
- [x] Code formatting: Run `uvx ruff format packages/`
- [x] Review: Verify point annotation API works via Swagger UI at /docs

**Phase 2 Complete:** Backend fully supports point annotations with CRUD API and SAM3 point-based inference. All tests pass, ready for frontend implementation.

---

## Phase 3: Frontend Point Annotation Page

**Goal:** Create interactive point annotation page with point placement and visualization

**Deliverable:** Working Point Annotation page where users can place/remove points on images

**Tasks:**

- [ ] [P3.1] Add point annotation API functions to `packages/samui-frontend/src/samui_frontend/api.py`
  - `create_point_annotation(image_id, x, y, is_positive)` → POST /point-annotations
  - `fetch_point_annotations(image_id)` → GET /point-annotations/{image_id}
  - `delete_point_annotation(annotation_id)` → DELETE /point-annotations/{annotation_id}

- [ ] [P3.2] Create `packages/samui-frontend/src/samui_frontend/components/point_annotator.py`
  - Use `streamlit_image_coordinates` for click detection
  - Draw existing points as small filled circles on image (green=positive, red=negative)
  - Return click coordinates when user clicks
  - Handle point hit detection for delete mode (check if click is near existing point)

- [ ] [P3.3] Create `packages/samui-frontend/src/samui_frontend/pages/point_annotation.py`
  - Page layout matching annotation page pattern:
    - Navigation controls (prev/next) above image
    - Main image area with point annotator
    - Sidebar with controls
  - Session state: selected_image_index, interaction_mode (add/delete), point_type (positive/negative)
  - Sidebar controls:
    - Add/Delete mode toggle (radio buttons)
    - Positive/Negative type toggle (radio buttons, only shown in add mode)
    - Process button
    - Point count display
  - Image thumbnail gallery below main image

- [ ] [P3.4] Implement point creation flow in `point_annotation.py`
  - On click in add mode: create point annotation via API
  - Refresh point list after creation
  - Display updated points on image

- [ ] [P3.5] Implement point deletion flow in `point_annotation.py`
  - On click in delete mode: find nearest point within threshold
  - Delete point annotation via API if found
  - Refresh point list after deletion

- [ ] [P3.6] Add Point Annotation page to navigation in `packages/samui-frontend/src/samui_frontend/app.py`
  - Add page to st.navigation() list
  - Add POINT mode to session state initialization

- [ ] [P3.7] Manual test: Verify point placement and deletion works correctly
  - Place positive points (green circles appear)
  - Place negative points (red circles appear)
  - Switch to delete mode and remove points
  - Points persist after page navigation

**Checkpoints:**

- [ ] Code quality: Run `uvx ruff check packages/ --fix`
- [ ] Code formatting: Run `uvx ruff format packages/`
- [ ] Review: Verify point annotation UX matches mockup layout

**Phase 3 Complete:** Point Annotation page functional with interactive point placement. Users can add/remove positive and negative points. Ready for processing integration.

---

## Phase 4: Processing and Mask Display Integration

**Goal:** Connect point processing to frontend with mask overlay display

**Deliverable:** Complete point mode workflow: annotate → process → view results with overlay toggle and alpha control

**Tasks:**

- [ ] [P4.1] Create `packages/samui-frontend/src/samui_frontend/components/mask_overlay.py`
  - Accept original image and mask image
  - Toggle state: show_overlay (boolean)
  - Alpha slider: 0-100% with 50% default
  - Render original image OR original with mask overlay based on toggle
  - Use PIL to composite mask onto original with alpha

- [ ] [P4.2] Add job creation for POINT mode in `point_annotation.py`
  - Process button creates job with mode=POINT
  - Pass current image_id to job creation
  - Show processing indicator while job runs
  - Poll job status until complete

- [ ] [P4.3] Add result fetching and display in `point_annotation.py`
  - After job completes, fetch mask from result
  - Display mask using MaskOverlay component
  - Add toggle button for overlay on/off
  - Add alpha slider below image

- [ ] [P4.4] Add POINT mode to history display in `packages/samui-frontend/src/samui_frontend/pages/history.py`
  - Include POINT results in history listing
  - Display point count for POINT mode results

- [ ] [P4.5] Manual end-to-end test of complete workflow
  - Upload image
  - Navigate to Point Annotation page
  - Place positive and negative points
  - Click Process
  - Verify mask displays correctly
  - Toggle overlay on/off
  - Adjust alpha slider
  - Verify result appears in History page

- [ ] [P4.6] Run full test suite: `cd packages/samui-backend && uv run pytest ../../tests/ -v`

**Checkpoints:**

- [ ] Code quality: Run `uvx ruff check packages/ --fix`
- [ ] Code formatting: Run `uvx ruff format packages/`
- [ ] Review: Complete end-to-end workflow test, verify all three modes work

**Phase 4 Complete:** Point mode fully functional with interactive annotation, processing, and result display. Overlay toggle and alpha slider working. Feature complete and ready for use.
