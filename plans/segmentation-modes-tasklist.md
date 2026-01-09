# Segmentation Modes Tasklist

## Phase 1: Data Model Extensions

**Goal:** Extend database models to support multiple segmentation modes and annotation types.

**Deliverable:** Updated Annotation and ProcessingResult models with new fields, passing migrations.

**Tasks:**

- [x] [P1.1] Add `PromptType` enum to `models.py` with values: `SEGMENT`, `POSITIVE_EXEMPLAR`, `NEGATIVE_EXEMPLAR`
- [x] [P1.2] Add `AnnotationSource` enum to `models.py` with values: `USER`, `MODEL`
- [x] [P1.3] Add `SegmentationMode` enum to `models.py` with values: `INSIDE_BOX`, `FIND_ALL`
- [x] [P1.4] Add `prompt_type` field to `Annotation` model (default=SEGMENT for backward compatibility)
- [x] [P1.5] Add `source` field to `Annotation` model (default=USER for backward compatibility)
- [x] [P1.6] Add `text_prompt` field to `Image` model (nullable string for find-all mode)
- [x] [P1.7] Add `mode` field to `ProcessingResult` model (default=INSIDE_BOX for backward compatibility)
- [x] [P1.8] Change `ProcessingResult` unique constraint from `image_id` to `(image_id, mode)`
- [x] [P1.9] Update `schemas.py` with new enums and fields in request/response models
  - Add `PromptType`, `AnnotationSource`, `SegmentationMode` enums
  - Update `AnnotationCreate` with optional `prompt_type` field
  - Update `AnnotationResponse` with `prompt_type` and `source` fields
  - Update `ImageResponse` with `text_prompt` field
  - Update `ProcessRequest` with `mode` field
  - Update `ProcessingResultResponse` with `mode` field
- [x] [P1.10] Add endpoint to update image text_prompt in `routes/images.py`
  - `PATCH /images/{image_id}` accepting `{"text_prompt": "..."}`
- [x] [P1.11] Update `routes/annotations.py` to accept and return new fields
- [x] [P1.12] Write tests for new annotation fields in `test_api_annotations.py`
- [x] [P1.13] Run tests: `cd packages/samui-backend && uv run pytest ../../tests/test_api_annotations.py -v`

**Checkpoints:**

- [x] Code quality: Run `uvx ruff check packages/samui-backend/`
- [x] Code format: Run `uvx ruff format packages/samui-backend/ --check`
- [x] Review: Verify all model changes are backward compatible with defaults

**Phase 1 Complete:** Data models extended with prompt_type, source, mode fields. Existing data remains valid through defaults. API accepts and returns new fields.

---

## Phase 2: SAM3 Find-All Service

**Goal:** Implement SAM3 batched API integration for find-all segmentation mode.

**Deliverable:** Working `process_image_find_all()` method that accepts text/exemplars and returns discovered masks and boxes.

**Tasks:**

- [x] [P2.1] Add batched API imports to `sam3_inference.py`
  - `from sam3.train.data.sam3_image_dataset import InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint`
  - `from sam3.train.data.collator import collate_fn_api as collate`
  - `from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI`
  - `from sam3.eval.postprocessors import PostProcessImage`
  - `from sam3.model.utils.misc import copy_data_to_device`
- [x] [P2.2] Add `_create_transforms()` helper method returning configured transform pipeline
  - RandomResizeAPI(sizes=1008, max_size=1008, square=True)
  - ToTensorAPI()
  - NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
- [x] [P2.3] Add `_create_postprocessor()` helper method returning configured PostProcessImage
  - detection_threshold=0.5, convert_mask_to_rle=False, use_original_sizes_mask=True
- [x] [P2.4] Add `_create_datapoint()` helper method that builds DataPoint from image, text, and boxes
  - Accept: PIL image, optional text_prompt, optional list of (box, is_positive) tuples
  - Create SAMImage with image data and size
  - Create FindQueryLoaded with query_text and optional input_bbox/input_bbox_label
  - Boxes in XYXY pixel format (transforms handle conversion)
- [x] [P2.5] Implement `process_image_find_all()` method in SAM3Service
  - Parameters: image (PIL), text_prompt (str|None), exemplar_boxes (list of (bbox_xywh, is_positive))
  - Validate: at least text_prompt or exemplar_boxes must be provided
  - Build datapoint, apply transforms, collate, move to GPU
  - Run model forward pass, postprocess results
  - Return: tuple of (masks array, scores array, discovered_bboxes in xywh format)
- [x] [P2.6] Add `load_model()` changes if needed for batched API (may need different model build)
  - Check if `enable_inst_interactivity=True` affects batched API
  - Batched API may work with base model without inst_interactivity
- [x] [P2.7] Write unit tests for `process_image_find_all()` in `test_sam3_inference.py`
  - Test with text prompt only
  - Test with positive exemplar box only
  - Test with text + positive + negative boxes
  - Test error when neither text nor boxes provided
  - Mock DataPoint creation, transforms, model forward, postprocessor
- [x] [P2.8] Run tests: `cd packages/samui-backend && uv run pytest ../../tests/test_sam3_inference.py -v`

**Checkpoints:**

- [x] Code quality: Run `uvx ruff check packages/samui-backend/src/samui_backend/services/sam3_inference.py tests/test_sam3_inference.py`
- [x] Code format: Run `uvx ruff format packages/samui-backend/src/samui_backend/services/sam3_inference.py tests/test_sam3_inference.py --check`
- [x] Review: Verify method signature matches plan, error handling is complete

**Phase 2 Complete:** SAM3Service has working find-all method. Batched API components properly integrated. Method tested with mocks.

---

## Phase 3: Processing Route Updates

**Goal:** Update processing routes to support mode selection and handle find-all output.

**Deliverable:** Processing endpoint accepts mode parameter, routes to correct SAM3 method, creates annotations from find-all discoveries.

**Tasks:**

- [x] [P3.1] Update `ProcessRequest` schema to include `mode: SegmentationMode` field with default `INSIDE_BOX`
- [x] [P3.2] Update `_process_single_image()` to accept mode parameter
- [x] [P3.3] Add `_process_single_image_find_all()` function in `processing.py`
  - Load image text_prompt from database
  - Load exemplar annotations (POSITIVE_EXEMPLAR, NEGATIVE_EXEMPLAR) for image
  - Convert annotations to (bbox_xywh, is_positive) format
  - Call `sam3.process_image_find_all()`
  - For each discovered box: create Annotation with source=MODEL, prompt_type=SEGMENT
  - Save masks and COCO JSON
  - Create/update ProcessingResult with mode=FIND_ALL
- [x] [P3.4] Update `_process_images_background()` to route based on mode
  - If mode=INSIDE_BOX: use existing `_process_single_image()`
  - If mode=FIND_ALL: use new `_process_single_image_find_all()`
- [x] [P3.5] Update image validation in `start_processing()` based on mode
  - INSIDE_BOX: require annotations with prompt_type=SEGMENT
  - FIND_ALL: require text_prompt OR annotations with prompt_type in (POSITIVE_EXEMPLAR, NEGATIVE_EXEMPLAR)
- [x] [P3.6] Update `_is_already_processed()` to check mode-specific ProcessingResult
- [x] [P3.7] Update mask/export endpoints to accept optional mode parameter
  - `GET /process/mask/{image_id}?mode=inside_box`
  - Default to INSIDE_BOX for backward compatibility
- [x] [P3.8] Create `tests/test_api_processing.py` with mode-aware processing tests
  - Test inside_box mode (existing behavior)
  - Test find_all mode with text prompt
  - Test find_all mode with exemplar boxes
  - Test validation errors for missing prompts
  - Test ProcessingResult created with correct mode
- [x] [P3.9] Run tests: `cd packages/samui-backend && uv run pytest ../../tests/ -v`

**Checkpoints:**

- [x] Code quality: Run `uvx ruff check packages/samui-backend/src/samui_backend/routes/processing.py tests/test_api_processing.py`
- [x] Code format: Run `uvx ruff format packages/samui-backend/src/samui_backend/routes/processing.py tests/test_api_processing.py --check`
- [x] Review: Verify backward compatibility - existing inside_box flow unchanged

**Phase 3 Complete:** Processing API supports both modes. Find-all creates annotations from discoveries. Mode-specific results tracked independently.

---

## Phase 4: Frontend Annotation Page

**Goal:** Add mode toggle and right-click support to annotation page.

**Deliverable:** Users can switch modes, draw positive/negative exemplar boxes, enter text prompts.

**Tasks:**

- [x] [P4.1] Add `segmentation_mode` to Streamlit session_state in `annotation.py`
  - Default to "inside_box"
  - Persist across page navigation
- [x] [P4.2] Add mode toggle UI at top of annotation page
  - Radio buttons or segmented control: "Inside Box" / "Find All"
  - Display description of each mode
- [x] [P4.3] Update `_fetch_annotations()` to filter by prompt_type based on mode
  - inside_box mode: fetch prompt_type=SEGMENT
  - find_all mode: fetch prompt_type in (POSITIVE_EXEMPLAR, NEGATIVE_EXEMPLAR)
- [x] [P4.4] Add text prompt input field (visible only in find_all mode)
  - Text input below mode toggle
  - On change: call PATCH /images/{image_id} to save text_prompt
  - Load existing text_prompt when selecting image
- [x] [P4.5] Update `_create_annotation()` to include prompt_type based on mode and exemplar type
  - inside_box mode: always SEGMENT
  - find_all mode: use selected exemplar type from toggle
- [x] [P4.6] Add exemplar type toggle for find-all mode
  - Add session state for `exemplar_type` (positive/negative)
  - Add radio toggle visible only in find-all mode
  - Use selected type when creating annotation
  - (Changed from right-click approach - see plan changelog)
- [x] [P4.7] Update `_render_annotation_list()` to show prompt_type indicator
  - Show "+" icon for positive exemplar, "-" for negative, none for segment
  - Different colors for positive (green) vs negative (red) exemplars
- [x] [P4.8] Update annotation deletion to work with filtered view
- [x] [P4.9] Manual test: verify mode switching, text prompt saving, exemplar toggle

**Checkpoints:**

- [x] Code quality: Run `uvx ruff check packages/samui-frontend/src/samui_frontend/pages/annotation.py packages/samui-frontend/src/samui_frontend/components/bbox_annotator.py`
- [x] Code format: Run `uvx ruff format packages/samui-frontend/src/samui_frontend/pages/annotation.py packages/samui-frontend/src/samui_frontend/components/bbox_annotator.py --check`
- [x] Review: Test both modes manually, verify UI is intuitive

**Phase 4 Complete:** Annotation page supports both modes. Users can select positive/negative exemplars via toggle. Text prompts saved per image.

---

## Phase 5: Frontend Processing Page

**Goal:** Add mode toggle to processing page with mode-aware gallery and status.

**Deliverable:** Users can select processing mode, see relevant annotations, process batch with selected mode.

**Tasks:**

- [x] [P5.1] Add mode toggle UI to processing page (matching annotation page style)
- [x] [P5.2] Update `_fetch_images()` or add helper to include annotation counts per mode
  - Show badge: "3 boxes" for inside_box, "2 exemplars + text" for find_all
- [x] [P5.3] Update image gallery to show annotations for selected mode only
  - Use existing `image_renderer` pattern with mode parameter
  - Different styling for segment boxes vs exemplar boxes
- [x] [P5.4] Update "Process" button to send selected mode in request
- [x] [P5.5] Update status display to show mode being processed
- [x] [P5.6] Update results display to be mode-aware
  - Show which mode was used for each result
  - Allow viewing results from either mode if both exist
- [x] [P5.7] Add visual indicator for images without annotations for selected mode
  - Gray out or show "No annotations for this mode"
- [x] [P5.8] Manual test: full workflow for both modes
  - Upload image → annotate (both modes) → process (each mode) → view results

**Checkpoints:**

- [x] Code quality: Run `uvx ruff check packages/samui-frontend/src/samui_frontend/pages/processing.py packages/samui-frontend/src/samui_frontend/components/image_gallery.py`
- [x] Code format: Run `uvx ruff format packages/samui-frontend/src/samui_frontend/pages/processing.py packages/samui-frontend/src/samui_frontend/components/image_gallery.py --check`
- [x] Review: Test complete workflow, verify mode switching is smooth

**Phase 5 Complete:** Processing page fully supports both modes. Gallery shows mode-relevant annotations. Batch processing works with selected mode.

---

## Phase 6: Integration Testing and Polish

**Goal:** Verify end-to-end functionality and fix any integration issues.

**Deliverable:** Complete working feature with both modes fully functional.

**Tasks:**

- [ ] [P6.1] Run full test suite: `cd packages/samui-backend && uv run pytest ../../tests/ -v`
- [ ] [P6.2] Manual end-to-end test: inside_box workflow
  - Upload image → draw segment boxes → process → view mask → export COCO
- [ ] [P6.3] Manual end-to-end test: find_all workflow
  - Upload image → enter text prompt → optionally draw exemplars → process → view discoveries → verify annotations created
- [ ] [P6.4] Test mode switching preserves annotations
  - Draw boxes in both modes, switch back and forth, verify nothing lost
- [ ] [P6.5] Test processing both modes on same image
  - Process with inside_box → process with find_all → verify both results accessible
- [ ] [P6.6] Fix any bugs discovered during integration testing
- [ ] [P6.7] Update any error messages to be mode-aware

**Checkpoints:**

- [ ] Code quality: Run `uvx ruff check packages/`
- [ ] Code format: Run `uvx ruff format packages/ --check`
- [ ] Review: Verify feature complete per plan scope, no regressions

**Phase 6 Complete:** Both segmentation modes fully functional. End-to-end workflows tested. Feature ready for use.
