# Batch Processing Refactor

Draft exploring how to refactor SAM3 processing from sequential per-image to true batch processing.

## Current State

### Processing Flow
```
Job with N images
  └── for each image (sequential):
       └── process_single_image()
            └── SAM3 inference (1 image)
            └── Save mask/COCO
```

### SAM3 API Usage by Mode

| Mode | API Used | Batching Support |
|------|----------|------------------|
| INSIDE_BOX | `predict_inst(box=...)` | Single image only |
| FIND_ALL | DataPoint batch API | Multi-image capable |
| POINT | `predict_inst(point_coords=...)` | Single image only |

**Key insight:** `predict_inst()` operates on a single `inference_state` (one image). The DataPoint/batch API can process multiple images in one forward pass.

## Goal

Process all images in a job with a single (or minimal) forward passes through SAM3.

## Challenges

### 1. API Mismatch

`predict_inst()` (used for INSIDE_BOX and POINT) doesn't support multi-image batching. Options:

**Option A: Convert INSIDE_BOX to batch API**
- The batch API supports box prompts via `input_bbox` in `FindQueryLoaded`
- Would need to convert from `predict_inst(box=...)` to DataPoint with boxes
- Boxes use different coordinate formats (predict_inst: xyxy pixels, batch API: xyxy pixels → transformed to cxcywh normalized)

**Option B: Keep predict_inst for INSIDE_BOX, batch at image level**
- Process multiple images by calling `set_image()` and `predict_inst()` in sequence
- Still sequential but could potentially pipeline image loading

**Option C: Hybrid approach**
- Use batch API where possible (FIND_ALL, potentially INSIDE_BOX)
- Keep `predict_inst` for POINT mode (no batch equivalent)

### 2. Point Mode Limitation

The batch API (`FindQueryLoaded`) doesn't support point prompts. Points are only available via `predict_inst()`. This means POINT mode cannot be converted to batch processing.

### 3. Memory Constraints

True batching loads multiple images into GPU memory simultaneously. Need to:
- Estimate memory per image based on resolution
- Dynamically determine batch size
- Handle OOM gracefully with fallback to smaller batches

### 4. Result Association

When batching, need to track which results belong to which image:
- Use `coco_image_id` in `InferenceMetadata` to map results back
- Ensure correct mask/COCO files saved per image

## Proposed Architecture

### Option 1: Mode-Specific Services (Recommended)

Split into separate services with shared base:

```
SAM3ServiceBase (abstract)
  ├── load_model() / unload_model()
  ├── is_loaded property
  └── _model, _processor references

SAM3InsideBoxService(SAM3ServiceBase)
  └── process_batch(images: list[Image], annotations: list[list[BboxAnnotation]])
      # Uses batch DataPoint API with box prompts

SAM3FindAllService(SAM3ServiceBase)
  └── process_batch(images: list[Image], queries: list[FindAllQuery])
      # Uses batch DataPoint API (already structured for this)

SAM3PointService(SAM3ServiceBase)
  └── process_single(image: Image, points: list[Point])
      # Uses predict_inst - cannot batch multiple images
```

**Pros:**
- Clean separation of concerns
- Each service optimized for its mode's capabilities
- Clear which modes support batching

**Cons:**
- Multiple service classes
- Need to coordinate model loading (share model instance?)

### Option 2: Single Service with Batch Methods

Keep one service, add batch methods:

```
SAM3Service
  ├── process_image(image, bboxes) → masks  # Legacy single
  ├── process_image_batch(images, bboxes_per_image) → list[masks]  # New batch
  ├── process_image_find_all(...) → FindAllResult  # Legacy single
  ├── process_image_find_all_batch(...) → list[FindAllResult]  # New batch
  └── process_image_points(image, points, labels) → masks  # Cannot batch
```

**Pros:**
- Single service, simpler dependency injection
- Backward compatible

**Cons:**
- Service becomes large with many methods
- Mixed capabilities (some batch, some not)

### Option 3: Batch Processor Wrapper

Create a separate batch processor that uses existing SAM3Service:

```
SAM3BatchProcessor
  ├── __init__(sam3_service: SAM3Service)
  └── process_job_batch(job: ProcessingJob, images: list[Image], ...)
      # Internally batches where possible, falls back to sequential
```

**Pros:**
- No changes to existing SAM3Service
- Encapsulates batching logic separately

**Cons:**
- Another layer of abstraction
- May need to expose internal SAM3Service methods

## Implementation Changes

### Job Processor Changes

```python
# Current (sequential)
def process_job(job_id):
    for image_id in job.image_ids:
        process_single_image(...)

# Proposed (batch where possible)
def process_job(job_id):
    if mode == SegmentationMode.POINT:
        # Cannot batch - process sequentially
        for image_id in job.image_ids:
            process_single_image(...)
    else:
        # Batch process
        images = load_all_images(job.image_ids)
        annotations = get_all_annotations(job.image_ids, mode)
        results = sam3.process_batch(images, annotations)
        save_all_results(results)
```

### Batch Size Management

```python
def estimate_batch_size(images: list[Image], available_memory_gb: float) -> int:
    """Estimate safe batch size based on image resolutions."""
    # SAM3 resizes to 1008x1008, estimate ~500MB per image
    memory_per_image_gb = 0.5
    max_batch = int(available_memory_gb / memory_per_image_gb)
    return min(max_batch, len(images))

def process_in_batches(images, batch_size):
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        yield process_batch(batch)
```

## Recommendation

**Start with Option 1 (Mode-Specific Services)** because:

1. POINT mode fundamentally cannot batch - clean separation makes this explicit
2. INSIDE_BOX conversion to batch API is non-trivial (coordinate transforms)
3. FIND_ALL is already structured for batch API
4. Easier to optimize each mode independently

**Phase the work:**

1. **Phase 1:** Extract `SAM3FindAllService` with true batching
2. **Phase 2:** Create `SAM3InsideBoxService` using batch API with boxes
3. **Phase 3:** Keep `SAM3PointService` as sequential (document limitation)
4. **Phase 4:** Update job processor to use batch methods

## Questions to Resolve

1. Should services share a model instance, or each load their own?
   - Sharing saves GPU memory but complicates lifecycle
   - Separate instances are simpler but wasteful

2. What's the target batch size? Fixed or dynamic?
   - Dynamic based on available GPU memory is more robust
   - Fixed is simpler to implement and test

3. Should we support mixed-mode jobs?
   - Currently a job has one mode for all images
   - Batching is simpler with single-mode jobs

4. Error handling in batch mode?
   - If one image fails, should the whole batch fail?
   - Or save partial results and continue?
