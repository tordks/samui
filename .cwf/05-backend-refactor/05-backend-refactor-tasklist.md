# Backend Refactor Tasklist

## Phase 1: Extract Database Helpers

**Goal:** Create reusable database query helpers to eliminate repeated patterns across routes.

**Deliverable:** Working `db/helpers.py` module with routes updated to use it.

**Tasks:**

- [x] [P1.1] Create `db/helpers.py` with `get_image_or_404(db, image_id)` function
  - Query Image by id, raise HTTPException(404) if not found
  - Return Image instance
- [x] [P1.2] Add `get_latest_processing_result(db, image_id, mode)` to `db/helpers.py`
  - Query ProcessingResult filtered by image_id and mode
  - Order by processed_at desc, return first or None
- [x] [P1.3] Update `db/__init__.py` to export helpers
- [x] [P1.4] Update `routes/images.py` to use `get_image_or_404`
  - Replace inline queries in `get_image`, `update_image`, `get_image_data`, `delete_image`, `get_image_history`
- [x] [P1.5] Update `routes/annotations.py` to use `get_image_or_404`
- [x] [P1.6] Update `routes/processing.py` to use helpers
- [x] [P1.7] Update `routes/jobs.py` to use `get_image_or_404`
  - Note: jobs.py batch validation pattern intentionally not changed (filters missing images rather than 404)
- [x] [P1.8] Run tests: `cd packages/samui-backend && uv run pytest ../../tests/ -v`

**Checkpoints:**

- [x] Code quality: Run `uvx ruff check --fix packages/samui-backend/`
- [x] Code quality: Run `uvx ruff format packages/samui-backend/`
- [x] Review: Verify all image existence checks now use helper, no duplicated query patterns remain

**Phase 1 Complete:** Database helpers extracted and integrated. Routes use shared query functions.

---

## Phase 2: Extract Annotation Snapshots Module

**Goal:** Extract annotation snapshot logic from job_processor into dedicated module.

**Deliverable:** Working `annotation_snapshots.py` with job_processor updated to import from it.

**Tasks:**

- [x] [P2.1] Create `services/annotation_snapshots.py` with functions extracted from `job_processor.py`:
  - `get_annotations_for_mode(db, image_id, mode)` (lines 32-52)
  - `get_point_annotations_for_image(db, image_id)` (lines 55-57)
  - `build_annotations_snapshot(db, image, mode)` (lines 60-94)
  - `_get_snapshot_annotation_ids(snapshot, mode)` (lines 97-102)
  - `_check_image_needs_processing(...)` (lines 105-138)
  - `filter_images_needing_processing(db, snapshots, mode)` (lines 141-173)
- [x] [P2.2] Update `services/__init__.py` to export annotation_snapshots functions
- [x] [P2.3] Update `job_processor.py` to import from `annotation_snapshots`
  - Remove duplicated function definitions
  - Update all call sites to use imported functions
- [x] [P2.4] Run tests: `cd packages/samui-backend && uv run pytest ../../tests/ -v`

**Checkpoints:**

- [x] Code quality: Run `uvx ruff check --fix packages/samui-backend/`
- [x] Code quality: Run `uvx ruff format packages/samui-backend/`
- [x] Review: Verify job_processor.py no longer contains snapshot logic, imports work correctly

**Phase 2 Complete:** Snapshot management extracted. job_processor.py reduced by ~140 lines.

---

## Phase 3: Extract Mode Processors

**Goal:** Extract mode-specific image processing functions from job_processor.

**Deliverable:** Working `mode_processors.py` with job_processor focused on orchestration.

**Tasks:**

- [ ] [P3.1] Create `services/mode_processors.py` with functions extracted from `job_processor.py`:
  - `_save_mask_to_storage(storage, masks, result_id)` (lines 176-198)
  - `_save_coco_to_storage(storage, image, bboxes, masks, result_id, points)` (lines 201-222)
  - `process_inside_box(storage, sam3, image, pil_image, bbox_annotations, result)` (lines 225-247)
  - `process_find_all(storage, sam3, image, pil_image, bbox_annotations, text_prompt, result)` (lines 250-298)
  - `process_point(storage, sam3, image, pil_image, point_annotations, result)` (lines 301-333)
- [ ] [P3.2] Update `services/__init__.py` to export mode_processors functions
- [ ] [P3.3] Update `job_processor.py` to import from `mode_processors`
  - Remove duplicated function definitions
  - Update `process_single_image` to use imported functions
- [ ] [P3.4] Run tests: `cd packages/samui-backend && uv run pytest ../../tests/ -v`

**Checkpoints:**

- [ ] Code quality: Run `uvx ruff check --fix packages/samui-backend/`
- [ ] Code quality: Run `uvx ruff format packages/samui-backend/`
- [ ] Review: Verify job_processor.py is now ~300 lines focused on job orchestration

**Phase 3 Complete:** Mode processors extracted. job_processor.py at target size (~300 lines).

---

## Phase 4: Extract SAM3 Batched Helpers

**Goal:** Extract batched API helpers from sam3_inference.py for find-all mode.

**Deliverable:** Working `sam3_batched.py` with sam3_inference.py focused on core inference.

**Tasks:**

- [ ] [P4.1] Create `services/sam3_batched.py` with methods extracted from `SAM3Service`:
  - `create_transforms()` (lines 180-199)
  - `create_postprocessor(detection_threshold)` (lines 201-220)
  - `create_datapoint(image, text_prompt, exemplar_boxes)` (lines 222-288)
  - `normalize_mask_output(masks_data, height, width)` (lines 290-319)
  - `boxes_xyxy_to_xywh(boxes_tensor)` (lines 321-326)
  - Move `DEFAULT_VISUAL_QUERY` constant
- [ ] [P4.2] Update `services/__init__.py` to export sam3_batched functions
- [ ] [P4.3] Update `sam3_inference.py` to import from `sam3_batched`
  - Remove method definitions from SAM3Service class
  - Update `process_image_find_all` to use imported functions
- [ ] [P4.4] Run tests: `cd packages/samui-backend && uv run pytest ../../tests/ -v`

**Checkpoints:**

- [ ] Code quality: Run `uvx ruff check --fix packages/samui-backend/`
- [ ] Code quality: Run `uvx ruff format packages/samui-backend/`
- [ ] Review: Verify sam3_inference.py is now ~280 lines focused on model and inference

**Phase 4 Complete:** SAM3 batched helpers extracted. sam3_inference.py at target size (~280 lines).

---

## Phase 5: Fix History Endpoint

**Goal:** Update history endpoint to include `text_prompt_used` and `point_count` from job data.

**Deliverable:** History API returns complete data that frontend expects.

**Tasks:**

- [ ] [P5.1] Update `ProcessingHistoryResponse` in `schemas.py`
  - Add `text_prompt_used: str | None = None` field
  - Add `point_count: int | None = None` field
- [ ] [P5.2] Update `get_image_history` in `routes/images.py`
  - Join ProcessingResult with ProcessingJob via job_id
  - Extract text_prompt and point_count from job.annotations_snapshot
  - Return enriched response objects
- [ ] [P5.3] Update `get_all_history` in `routes/images.py` with same join logic
- [ ] [P5.4] Run tests: `cd packages/samui-backend && uv run pytest ../../tests/ -v`
- [ ] [P5.5] Verify frontend history page displays text_prompt and point_count labels

**Checkpoints:**

- [ ] Code quality: Run `uvx ruff check --fix packages/samui-backend/`
- [ ] Code quality: Run `uvx ruff format packages/samui-backend/`
- [ ] Review: Verify API returns text_prompt_used and point_count, frontend displays them

**Phase 5 Complete:** History endpoint fixed. Frontend displays full result metadata.

---

## Final Validation

After all phases complete:

- [ ] Run full test suite: `cd packages/samui-backend && uv run pytest ../../tests/ -v`
- [ ] Run vulture to check for dead code: `uvx vulture packages/samui-backend/src/`
- [ ] Verify file sizes meet targets:
  - `job_processor.py` ~300 lines
  - `sam3_inference.py` ~280 lines
- [ ] Manual smoke test: upload image, add annotations, run processing, check history
