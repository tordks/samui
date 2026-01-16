# Point Mode Plan

Point-based segmentation mode with SAM1-style interactive input for SAM3 WebUI.

---

## Overview

### Problem

Users can only segment images using bounding boxes (Inside Box mode) or text prompts with exemplar boxes (Find All mode). There is no support for SAM1-style point-based interaction where users click directly on objects to indicate what to segment. Point prompts offer more precise control for certain segmentation tasks.

Additionally, the current `Annotation` table mixes bbox fields with different prompt types, making it awkward to add point coordinates.

### Purpose

Add a new Point Mode for interactive point-based segmentation:
- Users click on images to place positive (foreground) and negative (background) points
- Points persist in the database for session continuity
- Processing creates masks using SAM3's point prompt API
- Results display with toggleable mask overlay and adjustable alpha

As a prerequisite, refactor the annotation data model to cleanly separate bbox and point annotations.

### Scope

**IN scope:**
- Split `Annotation` table into `BboxAnnotation` and `PointAnnotation`
- New `PointAnnotation` model with x, y coordinates and positive/negative label
- New Point Annotation page with interactive point placement
- Add/Delete mode toggle for point management
- Positive/Negative point type toggle
- SAM3 point-based inference using `predict_inst()` with `point_coords`/`point_labels`
- Mask overlay toggle (original vs overlay) with alpha slider
- Full persistence: points as annotations, results as ProcessingResults
- Job-based processing matching existing pattern

**OUT of scope:**
- Auto-process on point change (may be added later)
- Modifier keys for quick positive/negative switching (toggle only)
- Point labels/numbers on display (color-coded only)
- Multi-mask output selection
- Undo/redo functionality

### Success Criteria

- Users can place positive and negative points on images via clicks
- Points persist across page refreshes and restarts
- Processing produces accurate masks from point prompts
- Mask overlay toggle and alpha slider work correctly
- All existing modes (Inside Box, Find All) continue working after refactor
- Test coverage matches existing patterns
- Zero regressions in existing functionality

---

## Solution Design

### System Architecture

**Core Components:**

- **BboxAnnotation Model:** Stores bounding box annotations (SEGMENT, POSITIVE_EXEMPLAR, NEGATIVE_EXEMPLAR)
- **PointAnnotation Model:** Stores point annotations with x, y coordinates and positive/negative label
- **Point Annotation Page:** New Streamlit page for interactive point-based segmentation
- **PointAnnotator Component:** Interactive click handler for placing/removing points on images
- **SAM3Service.process_image_points():** New method for point-based inference
- **Updated JobProcessor:** Extended to handle POINT mode with point annotations

**Project Structure:**

```
packages/samui-backend/src/samui_backend/
├── db/
│   └── models.py [MODIFY] - Split Annotation → BboxAnnotation + PointAnnotation
├── routes/
│   └── annotations.py [MODIFY] - Update for BboxAnnotation, add point annotation endpoints
├── services/
│   ├── sam3_inference.py [MODIFY] - Add process_image_points() method
│   ├── job_processor.py [MODIFY] - Handle POINT mode
│   └── coco_export.py [MODIFY] - Support point annotations in export
├── schemas.py [MODIFY] - Add point annotation schemas
└── enums.py [MODIFY] - Add POINT mode

packages/samui-frontend/src/samui_frontend/
├── pages/
│   ├── annotation.py [MODIFY] - Update for BboxAnnotation API
│   └── point_annotation.py [CREATE] - New point-based annotation page
├── components/
│   ├── point_annotator.py [CREATE] - Interactive point placement component
│   └── mask_overlay.py [CREATE] - Toggleable mask overlay with alpha control
├── api.py [MODIFY] - Add point annotation endpoints
└── app.py [MODIFY] - Add Point Annotation page to navigation

tests/
├── test_api_annotations.py [MODIFY] - Update for split models
├── test_api_point_annotations.py [CREATE] - Point annotation API tests
├── test_sam3_inference.py [MODIFY] - Add point inference tests
└── test_job_processor.py [MODIFY] - Add point mode processing tests
```

**Component Relationships:**

- PointAnnotation depends on Image (foreign key relationship)
- BboxAnnotation depends on Image (foreign key relationship)
- Point Annotation Page uses PointAnnotator component for interaction
- Point Annotation Page uses MaskOverlay component for result display
- JobProcessor queries PointAnnotation for POINT mode processing
- SAM3Service.process_image_points() called by JobProcessor for POINT mode

**Relationship to Existing Codebase:**

- Architectural layer: Extends existing annotation and processing patterns
- Follows: Repository's service-oriented architecture and dependency injection
- Uses: Existing job queue system, storage service, SAM3 model loading
- Extends: Current segmentation modes (INSIDE_BOX, FIND_ALL) with new POINT mode
- Mirrors: Annotation page patterns for navigation, image selection, and mode toggling

### Design Rationale

**Split Annotation into BboxAnnotation and PointAnnotation**

Cleaner data model with no nullable fields. Each table stores only relevant fields:
- BboxAnnotation: x, y, width, height, prompt_type
- PointAnnotation: x, y, is_positive

Alternative considered: Single polymorphic table with nullable point_x, point_y fields. Rejected because it creates messy schema where half the columns are NULL depending on annotation type.

**Dedicated Point Annotation page vs. integrating into existing Annotation page**

Point mode has different interaction model:
- Tighter annotation-result loop (annotate AND view results on same page)
- Toggle between original and overlay
- Alpha slider for mask transparency

Separate page allows optimized UX without complicating existing annotation page logic.

**Add/Delete mode toggle for point management**

Simple two-mode interaction:
- Add mode: clicks create points (positive or negative based on type toggle)
- Delete mode: clicks on existing points remove them

Alternative considered: Click on existing point to delete (needs hit detection). Rejected for simpler implementation and clearer UX.

**Visual mockup:** See `.cwf/point-mode/point-mode-mockup.html` for Point Annotation page layout.

### Technical Specification

**Dependencies:**

Required (existing):
- SQLAlchemy 2.0+ (database ORM)
- FastAPI 0.100+ (API framework)
- Streamlit 1.30+ (frontend framework)
- streamlit-image-coordinates (click detection)
- PyTorch 2.7+ with CUDA (SAM3 inference)
- Pillow (image processing)

No new dependencies required.

**Runtime Behavior:**

Point Annotation Flow:
1. User selects image from gallery or navigation
2. User toggles to Add mode (default) with Positive type (default)
3. User clicks on image → point created at click coordinates
4. Point saved to PointAnnotation table via API
5. Point displayed as small colored circle (green=positive, red=negative)
6. User can toggle to Delete mode and click points to remove
7. User clicks Process → job created with POINT mode
8. JobProcessor loads SAM3, calls process_image_points()
9. Mask saved to storage, ProcessingResult created
10. Frontend fetches mask, displays with overlay toggle and alpha slider

SAM3 Point Inference:
```python
def process_image_points(
    self,
    image: Image.Image,
    points: list[tuple[int, int]],
    labels: list[int]  # 1=positive, 0=negative
) -> NDArray[np.uint8]:
    # Set image in processor
    inference_state = self._processor.set_image(image_array)

    # Convert to numpy arrays
    point_coords = np.array(points)  # shape: (N, 2)
    point_labels = np.array(labels)  # shape: (N,)

    # Call predict_inst with point prompts
    masks, scores, _ = self._model.predict_inst(
        inference_state,
        point_coords=point_coords,
        point_labels=point_labels,
        box=None,
        multimask_output=False,
    )

    return masks[0]  # Single combined mask
```

**Error Handling:**

- No points selected → 400 "No point annotations found for this image"
- Point outside image bounds → 400 "Point coordinates must be within image dimensions"
- Processing fails → Job marked FAILED with error message
- SAM3 model not loaded → Load on demand (existing pattern)

**Configuration:**

No new configuration required. Uses existing:
- DATABASE_URL for PostgreSQL
- AZURE_STORAGE_CONNECTION_STRING for blob storage
- HF_TOKEN for SAM3 model access

---

## Implementation Strategy

### Development Approach

**Foundation First with Safe Increments**

Build in phases that each produce working, testable code:

1. **Data Model Refactor:** Split annotations first. This is prerequisite work that establishes clean foundation. All existing tests must pass after this phase.

2. **Backend Point Support:** Add PointAnnotation model, API routes, and SAM3 point inference. Backend fully functional before frontend work.

3. **Frontend Point Page:** Build new page with interactive point placement, leveraging existing navigation patterns.

4. **Integration:** Connect frontend to backend, add mask overlay display with toggle and alpha control.

### Testing Approach

**Integration-focused matching existing patterns:**

- API integration tests for new point annotation endpoints
- SAM3 service tests with mocked model for process_image_points()
- Job processor tests for POINT mode handling
- Update existing tests for BboxAnnotation rename

Tests use in-memory SQLite and mocked storage (existing fixtures).

### Risk Mitigation

- **Regression risk:** Run full test suite after Phase 1 to ensure existing modes work
- **SAM3 API compatibility:** Verify point_coords/point_labels format matches SAM3 expectations early

### Checkpoint Strategy

Each phase ends with validation:
- **Self-review:** Agent reviews implementation against phase deliverable
- **Code quality:** Run `uvx ruff check packages/ --fix` and `uvx ruff format packages/`
- **Tests:** Run `uv run pytest ../../tests/ -v` from backend package

These checkpoints ensure code quality before proceeding to next phase.

---

## Changelog

### Phase 2 Amendment (during implementation)

**Refactor `process_single_image` into mode-specific helpers**

The original plan added POINT mode as another branch in the monolithic `process_single_image` function. During implementation, we identified this would make the function harder to maintain with three modes.

**Decision:** Extract mode-specific processing into helper functions:
- `_process_inside_box(db, storage, sam3, image, result)` - bbox inference
- `_process_find_all(db, storage, sam3, image, result)` - find-all inference
- `_process_point(db, storage, sam3, image, result)` - point inference

Keep `process_single_image` as orchestrator handling shared setup (load image, create result) and teardown (commit).

**Rationale:** Better separation of concerns, each mode's logic is isolated and testable, easier to add future modes.
