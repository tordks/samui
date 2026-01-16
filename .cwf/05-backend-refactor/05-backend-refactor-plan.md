# Backend Refactor Plan

## Overview

### Problem

Two backend service files have grown beyond maintainable size:
- `job_processor.py` (528 lines) - mixes job orchestration, mode-specific processing, snapshot management, and annotation filtering
- `sam3_inference.py` (416 lines) - combines model management, three inference modes, and batched API helpers

Additionally, repeated query patterns appear across route files (image existence checks, processing result queries), and the history API endpoint doesn't provide `text_prompt_used` or `point_count` that the frontend expects.

### Purpose

Improve backend maintainability by:
1. Splitting large service files into focused modules
2. Extracting common database query patterns into reusable helpers
3. Fixing the history API to include job-level data the frontend needs

### Scope

**IN scope:**
- Split `job_processor.py` into smaller, focused modules
- Split `sam3_inference.py` by extracting batched API helpers
- Create database helper functions for repeated query patterns
- Update history endpoint to include `text_prompt_used` and `point_count` from job data

**OUT of scope:**
- Service singleton pattern changes (deferred - works adequately)
- Shared enum package between frontend/backend (keep duplication)
- Frontend changes (already handles missing fields gracefully)

### Success Criteria

- `job_processor.py` reduced to ~300 lines (core orchestration only)
- `sam3_inference.py` reduced to ~280 lines (core inference only)
- No repeated image/result query patterns in route files
- History endpoint returns `text_prompt_used` and `point_count`
- All existing tests pass
- No API contract changes (except history endpoint additions)

---

## Solution Design

### System Architecture

**Core Components:**

- **JobProcessor (refactored):** Core job orchestration - queue management, job lifecycle, image iteration
- **ModeProcessors:** Mode-specific processing logic extracted from job_processor
  - `InsideBoxProcessor` - bbox prompt processing
  - `FindAllProcessor` - text/exemplar discovery processing
  - `PointProcessor` - point prompt processing
- **SnapshotManager:** Annotation snapshot building and comparison logic
- **SAM3Service (refactored):** Core model management and inference methods
- **SAM3BatchedHelpers:** Transform pipeline, postprocessor, datapoint creation for find-all mode
- **DatabaseHelpers:** Common query patterns (`get_image_or_404`, `get_latest_processing_result`)

**Project Structure:**

```
packages/samui-backend/src/samui_backend/
├── db/
│   ├── __init__.py [MODIFY]
│   ├── database.py
│   ├── models.py
│   └── helpers.py [CREATE]
├── routes/
│   ├── __init__.py
│   ├── annotations.py [MODIFY]
│   ├── images.py [MODIFY]
│   ├── jobs.py [MODIFY]
│   └── processing.py [MODIFY]
├── services/
│   ├── __init__.py [MODIFY]
│   ├── coco_export.py
│   ├── job_processor.py [MODIFY]
│   ├── mode_processors.py [CREATE]
│   ├── snapshot_manager.py [CREATE]
│   ├── sam3_inference.py [MODIFY]
│   ├── sam3_batched.py [CREATE]
│   └── storage.py
├── schemas.py [MODIFY]
└── ...
```

**Component Relationships:**

- `job_processor.py` imports from `mode_processors.py` and `snapshot_manager.py`
- `mode_processors.py` imports from `sam3_inference.py` and `storage.py`
- `sam3_inference.py` imports from `sam3_batched.py` for find-all mode
- Route files import from `db/helpers.py` for common queries
- `routes/images.py` joins through `ProcessingJob` for history data

**Relationship to Existing Codebase:**

- Architectural layer: Service layer refactoring (internal restructure)
- All public APIs remain unchanged except history endpoint additions
- Follows existing patterns: functions over classes for processors (matches current style)
- Uses existing dependency injection via FastAPI Depends

---

### Design Rationale

**Extract mode processors as functions, not classes**

The existing `_process_inside_box`, `_process_find_all`, `_process_point` functions are already well-defined. Moving them to a separate module preserves the current functional style used throughout the codebase.

Alternative considered:
- Processor classes with common interface - adds abstraction overhead without clear benefit for three fixed modes

**Keep snapshot logic in dedicated module**

Snapshot building and comparison logic (lines 60-173 in job_processor.py) forms a cohesive unit around a single concept. Extracting it makes job_processor focused on orchestration.

**Extract SAM3 batched helpers vs keeping inline**

The `_create_transforms`, `_create_postprocessor`, `_create_datapoint`, `_normalize_mask_output` methods are only used by `process_image_find_all`. Extracting them:
- Reduces SAM3Service to core responsibilities (load/unload/inference)
- Makes batched API complexity isolated and testable

Trade-off accepted:
- Slightly more files to navigate
- Clearer separation of interactive vs batched inference paths

**Database helpers as functions, not a repository class**

Route files need simple query helpers, not a full repository pattern. Functions like `get_image_or_404(db, image_id)` match FastAPI's functional style and are easy to understand.

---

### Technical Specification

**Dependencies:**

No new dependencies required. All changes use existing libraries:
- SQLAlchemy (database queries)
- Pydantic (schema updates)
- FastAPI (route dependencies)

**Runtime Behavior:**

No runtime behavior changes for existing functionality. The refactoring is purely structural.

History endpoint change:
1. Query `ProcessingResult` with join to `ProcessingJob`
2. Extract `text_prompt` from job's `annotations_snapshot[image_id].text_prompt`
3. Count point annotations from job's `annotations_snapshot[image_id].point_annotations`
4. Return as computed fields in `ProcessingHistoryResponse`

**File Size Targets:**

| File | Current | Target | Reduction |
|------|---------|--------|-----------|
| `job_processor.py` | 528 | ~300 | ~43% |
| `sam3_inference.py` | 416 | ~280 | ~33% |
| `mode_processors.py` | - | ~150 | new |
| `snapshot_manager.py` | - | ~100 | new |
| `sam3_batched.py` | - | ~140 | new |
| `db/helpers.py` | - | ~50 | new |

---

## Implementation Strategy

### Development Approach

**Bottom-up with safe increments:**

1. Create new modules with extracted code first (no existing code changes)
2. Update imports and remove duplicated code
3. Each phase produces working, testable code
4. History endpoint fix is independent and done last

This approach minimizes risk - new modules can be tested before removing original code.

### Testing Approach

- Run existing test suite after each phase to catch regressions
- No new unit tests required (extracting existing tested code)
- History endpoint change may need a test update if there's an existing test

### Checkpoint Strategy

Each phase ends with validation:
- **Self-review:** Verify extracted code matches original behavior
- **Code quality:** `uvx ruff check --fix packages/samui-backend/`
- **Tests:** `cd packages/samui-backend && uv run pytest ../../tests/ -v`

---
