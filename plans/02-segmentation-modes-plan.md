# Segmentation Modes Plan

## Overview

### Problem

The current SAM3 integration only supports one segmentation workflow: users draw bounding boxes and the model segments objects **inside** each box. This limits usability for scenarios where users want to quickly find all instances of an object type across an image using text descriptions or visual exemplars.

### Purpose

Add a second segmentation mode ("find-all") that uses SAM3's batched API to discover all instances matching a text prompt and/or visual exemplars. Users can switch between "inside-box" mode (current behavior) and "find-all" mode, with each mode having its own annotations and processing results per image.

### Scope

**IN scope:**

- Find-all segmentation mode using SAM3 batched API
- Text prompt input per image for find-all mode
- Positive/negative exemplar boxes (left-click = positive, right-click = negative)
- Mode toggle on Annotation and Processing pages
- Per-mode annotation storage and display
- Per-mode processing results
- Auto-creation of annotations from find-all discoveries (marked as model-generated)

**OUT of scope:**

- Multi-mask output toggle (returns multiple mask candidates per box)
- Interactive/point-prompt segmentation workflow
- Real-time mask preview during annotation
- Batch-level text prompts (each image has its own)

### Success Criteria

- Users can switch between inside-box and find-all modes on both Annotation and Processing pages
- Find-all mode accepts text prompts and optional positive/negative exemplar boxes
- Right-click drag creates negative exemplar boxes in find-all mode
- Gallery displays only annotations for the currently selected mode
- Processing respects selected mode for entire batch
- Discovered objects from find-all are saved as annotations with `source=model`
- Each mode tracks its own processing status independently
- All new functionality covered by mocked tests matching existing test style

---

## Solution Design

### System Architecture

**Core Components:**

- **Annotation Model Extension**: Stores `prompt_type` (segment/positive_exemplar/negative_exemplar) and `source` (user/model) per annotation
- **ProcessingResult Extension**: Stores `mode` field, unique constraint on (image_id, mode) to allow multiple results per image
- **SAM3Service.process_image_find_all()**: New method using batched API (DataPoint, FindQueryLoaded, transforms, postprocessor)
- **Processing Route**: Accepts mode parameter, routes to appropriate SAM3 method
- **Frontend Mode Toggle**: Shared state for mode selection, filters annotations by mode
- **BBox Annotator**: Extended to support right-click drag for negative boxes

**Project Structure:**

```
packages/samui-backend/src/samui_backend/
├── db/
│   └── models.py [MODIFY] - Add PromptType enum, source field, ProcessingResult.mode
├── schemas.py [MODIFY] - Add SegmentationMode enum, update request/response schemas
├── services/
│   └── sam3_inference.py [MODIFY] - Add process_image_find_all() method
├── routes/
│   └── processing.py [MODIFY] - Accept mode, route to correct method, handle find-all output
│   └── annotations.py [MODIFY] - Support new annotation fields

packages/samui-frontend/src/samui_frontend/
├── components/
│   └── bbox_annotator.py [MODIFY] - Right-click drag support for negative boxes
│   └── image_gallery.py [MODIFY] - Mode-aware annotation display
├── pages/
│   └── annotation.py [MODIFY] - Mode toggle, filtered annotation display, text prompt input
│   └── processing.py [MODIFY] - Mode toggle, filtered gallery, mode-aware status display

tests/
├── test_sam3_inference.py [MODIFY] - Add tests for process_image_find_all()
├── test_api_annotations.py [MODIFY] - Test new annotation fields
├── test_api_processing.py [CREATE] - Test mode-aware processing
```

**Component Relationships:**

- Frontend pages depend on shared mode state (Streamlit session_state)
- Processing route depends on SAM3Service methods (selects based on mode)
- SAM3Service find-all method depends on batched API imports (DataPoint, transforms, postprocessor)
- Annotation display depends on prompt_type filtering
- Processing status derived from ProcessingResult records per mode

**Relationship to Existing Codebase:**

- Extends existing `Annotation` model with new fields (non-breaking, nullable/default values)
- Extends existing `ProcessingResult` model, changes unique constraint
- Adds new SAM3Service method alongside existing `process_image()`
- Frontend changes are additive (mode toggle, right-click handling)
- Follows existing patterns: SQLAlchemy models, Pydantic schemas, FastAPI routes, Streamlit pages

---

### Design Rationale

**Use batched API (DataPoint/FindQueryLoaded) for find-all mode**

The SAM3 `predict_inst` method only supports box/point prompts for single-object segmentation. Text prompts and exemplar-based discovery require the batched API which:
- Accepts text queries via `query_text` parameter
- Supports positive/negative exemplar boxes via `input_bbox` + `input_bbox_label`
- Returns all discovered instances, not just one per prompt

Alternative considered:
- Extend `predict_inst` usage: Not possible, API doesn't support text prompts

**Store prompt_type on Annotation rather than separate tables**

Single table with discriminator field is simpler than separate tables for each annotation type. All annotation types share the same core fields (image_id, bbox coordinates). The `prompt_type` field distinguishes semantics.

Trade-offs:
- Pro: Single query to fetch all annotations, simpler schema
- Con: Some fields may be unused depending on type (acceptable for this case)

**Per-mode ProcessingResult with (image_id, mode) uniqueness**

Allows an image to have independent processing results for each mode. Status is derived by checking which results exist rather than maintaining separate status fields.

Alternative considered:
- Separate status fields per mode on Image: More complex, requires migration of existing status logic

**Auto-create annotations from find-all discoveries**

Discovered objects become first-class annotations that users can edit/delete. The `source=model` field distinguishes them from user-drawn annotations for filtering/styling.

Alternative considered:
- Store discoveries separately: Would require duplicate UI for viewing/managing

---

### Technical Specification

**Dependencies:**

New imports required in sam3_inference.py (from sam3 package):
- `sam3.train.data.sam3_image_dataset`: InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint
- `sam3.train.data.collator`: collate_fn_api
- `sam3.train.transforms.basic_for_api`: ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
- `sam3.eval.postprocessors`: PostProcessImage
- `sam3.model.utils.misc`: copy_data_to_device

Existing dependencies (no changes):
- sam3, torch, numpy, PIL (backend)
- streamlit, httpx, PIL (frontend)

**Runtime Behavior:**

Inside-box mode (unchanged):
1. Load annotations with `prompt_type=segment`
2. Call `sam3.process_image(image, bboxes)` using `predict_inst`
3. Save masks and COCO JSON
4. Create/update ProcessingResult with `mode=inside_box`

Find-all mode (new):
1. Load text prompt for image (stored in new field or separate table)
2. Load annotations with `prompt_type` in (positive_exemplar, negative_exemplar)
3. Build DataPoint with image, text query, and exemplar boxes
4. Apply transforms (resize to 1008x1008, normalize)
5. Call model forward pass, postprocess results
6. For each discovered object:
   - Create Annotation with `prompt_type=segment`, `source=model`
   - Include discovered bbox coordinates
7. Save combined masks and COCO JSON
8. Create/update ProcessingResult with `mode=find_all`

**Text Prompt Storage:**

Add `text_prompt` field to Image model (nullable string). Set via new API endpoint or annotation page UI.

**Error Handling:**

- No text prompt and no exemplar boxes for find-all: Return 400 "Find-all requires text prompt or exemplar boxes"
- Find-all discovers no objects: Return success with empty results, no annotations created
- Model not loaded: Existing error handling applies

**Configuration:**

No new configuration required. Uses existing SAM3 model loading.

---

## Implementation Strategy

### Development Approach

**Bottom-up with vertical slices:**

1. **Foundation**: Extend data models first (Annotation, ProcessingResult) - enables all subsequent work
2. **Backend API**: Add SAM3 find-all method, then route changes - can be tested independently
3. **Frontend Integration**: Mode toggle and UI changes last - depends on backend being ready

Each phase produces testable, working code. Backend changes are deployed before frontend depends on them.

### Testing Approach

**Mocked unit tests matching existing style:**

- Mock SAM3 batched API components (DataPoint, transforms, postprocessor, model forward)
- Test process_image_find_all() returns expected structure
- Test annotation creation with new fields
- Test processing route mode selection logic

No integration tests against real SAM3 model (matches existing approach in test_sam3_inference.py).

### Checkpoint Strategy

Each phase ends with validation:
- **Self-review**: Verify implementation matches phase deliverable
- **Code quality**: Run `uvx ruff check packages/` and `uvx ruff format packages/ --check`
- **Tests**: Run `uv run pytest tests/ -v` from backend package

These ensure code quality before proceeding to dependent phases.

---

## Changelog

**2026-01-09**: Phase 4 implementation change
- **Original**: Right-click drag to create negative exemplar boxes in find-all mode
- **Changed to**: Toggle control (Positive/Negative) for selecting exemplar type before drawing
- **Reason**: The `streamlit-image-coordinates` component does not support right-click detection. A toggle provides clear user intent and is more accessible.

