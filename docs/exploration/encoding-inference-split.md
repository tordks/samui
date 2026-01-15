# SAM3 Encoding Cache Design

Summary from exploration with Claude

## Problem Statement

Currently, each inference call re-encodes the image through the ViT backbone (~2-3s). For interactive editing where users add/modify prompts, this creates unacceptable latency. We need to:

1. Encode images once, cache the encodings
2. Run inference multiple times with cached encodings (~200-500ms)
3. Support 50-100+ images with persistent storage
4. Scale to multiple SAM3 workers

## Design Principles

1. **Use existing API**: Cache the `state` dict from `processor.set_image()` directly
2. **Encode on upload**: Background encode when images are uploaded
3. **Two-tier storage**: Local SSD (fast) + Azure Blob (persistent)
4. **Portable format**: Use safetensors for fast, safe serialization

---

## Embedding Modes

SAM3 supports **two embedding modes** with different sizes:

| Mode | Flag | Levels | Size (bfloat16) | Use Case |
|------|------|--------|-----------------|----------|
| **SAM3 Full** | default | 4 FPN levels | ~107MB | Find-all, text prompts, batched |
| **SAM1-Style** | `enable_inst_interactivity=True` | 3 FPN levels | **~53MB** | Interactive box/point prompts |

### SAM1-Style Mode (Recommended for Frontend)

SAM3 has built-in SAM1-compatible inference via a **dual-neck architecture**:

```python
# Build model with SAM1-style support
model = build_sam3_image_model(enable_inst_interactivity=True)

# Backbone output contains BOTH embedding types:
state["backbone_out"] = {
    "backbone_fpn": [...],           # SAM3 full: 4 levels, ~107MB
    "sam2_backbone_out": {           # SAM1-style: 3 levels, ~53MB
        "backbone_fpn": [...],       # 288², 144², 72² only
    }
}
```

### Embedding Size Breakdown

| Embedding Type | Levels | Shape | Size (bfloat16) |
|----------------|--------|-------|-----------------|
| SAM3 Full | 288², 144², 72², 36² | 4 tensors | ~107MB |
| SAM1-Style | 288², 144², 72² | 3 tensors | ~53MB |
| Lowest only | 72² | 1 tensor | ~2.5MB |

---

## Latency Analysis

### Transfer Times (Azure Blob, 60 MB/s)

| Embedding | Size | Download Time |
|-----------|------|---------------|
| SAM3 Full (bfloat16) | 107MB | ~1.8s |
| **SAM1-Style (bfloat16)** | **53MB** | **~0.9s** |
| SAM1-Style (compressed) | ~30MB | ~0.5s |

### Why SAM1-Style Changes the Calculus

| Operation | SAM3 Full | SAM1-Style |
|-----------|-----------|------------|
| ViT encoding (GPU) | 2-3s | 2-3s |
| Azure Blob download | 1.8s | **0.9s** |
| Local SSD load | 300-600ms | 150-300ms |
| Inference (cached) | 200-500ms | 200-500ms |

**Key insight**: With SAM1-style embeddings (~53MB), Azure Blob download is **faster than recalculating**. Local SSD cache becomes optional optimization rather than requirement.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Upload Flow                                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Upload Image ──▶ Save to Blob ──▶ Queue Encoding Job               │
│                                           │                          │
│                                           ▼                          │
│                              ┌─────────────────────┐                 │
│                              │  Encoding Worker    │                 │
│                              │  (Background)       │                 │
│                              │                     │                 │
│                              │  1. Download image  │                 │
│                              │  2. Run ViT encoder │                 │
│                              │  3. Save to Blob    │                 │
│                              └─────────────────────┘                 │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                         Inference Flow                               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Inference Request                                                   │
│        │                                                             │
│        ▼                                                             │
│  ┌─────────────┐    Hit     ┌─────────────────┐                      │
│  │ Local SSD   │ ─────────▶ │ Run Inference   │ ──▶ Return masks     │
│  │ Cache       │            │ (~200-500ms)    │                      │
│  └─────────────┘            └─────────────────┘                      │
│        │ Miss                                                        │
│        ▼                                                             │
│  ┌─────────────┐            ┌─────────────────┐                      │
│  │ Azure Blob  │ ─────────▶ │ Cache to SSD    │ ──▶ Run Inference    │
│  │ (~3-4s)     │            │ then infer      │                      │
│  └─────────────┘            └─────────────────┘                      │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Storage Tiers

| Tier | Capacity | Latency | Purpose |
|------|----------|---------|---------|
| Local SSD | 50-100GB (~250-500 images) | 300-600ms | Hot cache per worker |
| Azure Blob | Unlimited | 3-4s | Persistent, shared storage |

---

## 1. Storage Options

### Option A: Azure Blob + Local SSD Cache

Best for: Backend-only inference, multiple workers, SAM3 full mode

| Layer | Throughput | Latency (53MB) | Latency (107MB) |
|-------|------------|----------------|-----------------|
| Local SSD | 500+ MB/s | 150-300ms | 300-600ms |
| Azure Blob | 60 MB/s | ~0.9s | ~1.8s |

```
Worker → Local SSD (hit?) → Return
                ↓ miss
         Azure Blob → Cache to SSD → Return
```

**Pros**: Cheap, simple blob storage
**Cons**: Cold start ~1-2s, local cache recommended for SAM3 full mode

---

### Option B: Azure Files Premium (Mounted)

Best for: Simplicity, consistent low latency, no local cache needed

| Metric | Value |
|--------|-------|
| Per-file throughput | 300 MB/s (up to 1 GB/s w/ SMB multichannel) |
| Latency | 2-3ms |
| Download 215MB | **~0.7s** |
| Cost | ~$0.15/GB/month provisioned |

```
Worker → NFS mount (/mnt/encodings) → Return (~700ms)
```

**Pros**: 5x faster than Blob, acts like local filesystem, no cache layer needed
**Cons**: Higher cost, provisioned capacity model

```python
# Mount Azure Files Premium as NFS
class AzureFilesPremiumStore(EncodingStore):
    def __init__(self, mount_path: Path = Path("/mnt/encodings")):
        self.mount_path = mount_path

    def load(self, image_id: UUID, device: str = "cuda") -> dict:
        return load_encoding(self.mount_path / f"{image_id}.safetensors", device)
```

---

### Option C: Azure Blob + Frontend Cache

Best for: React frontend, offline-capable, reduces backend load

| Operation | Latency | Notes |
|-----------|---------|-------|
| First load (Blob → Browser) | 3-4s | One-time per browser session |
| Subsequent loads (IndexedDB) | 50-200ms | Cached locally in browser |
| Inference (ONNX in browser) | 200-500ms | No backend needed |

```
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend Caching Flow                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Browser                          Backend                      │
│   ┌──────────────┐                ┌──────────────┐              │
│   │ IndexedDB    │  ── miss ──►   │ Azure Blob   │              │
│   │ Cache        │  ◄── 215MB ──  │ (encodings)  │              │
│   │ (~215MB/img) │                └──────────────┘              │
│   └──────────────┘                                              │
│         │                                                       │
│         ▼ hit (50-200ms)                                        │
│   ┌──────────────┐                                              │
│   │ ONNX Runtime │  ── inference ──►  Masks                     │
│   │ (WebGPU)     │                                              │
│   └──────────────┘                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Pros**:
- Cheap blob storage works fine (download once)
- Offline capable after first load
- Reduces backend GPU costs
- Better UX after initial cache

**Cons**:
- Requires ONNX decoder on frontend
- Large browser storage (~215MB per image)
- Initial 3-4s load per new image

```typescript
// Frontend encoding cache using IndexedDB
class EncodingCache {
  private db: IDBDatabase;

  async get(imageId: string): Promise<ArrayBuffer | null> {
    // Check IndexedDB first
    const cached = await this.db.get('encodings', imageId);
    if (cached) return cached;

    // Download from backend
    const response = await fetch(`/api/encodings/${imageId}`);
    const encoding = await response.arrayBuffer();

    // Cache for next time
    await this.db.put('encodings', encoding, imageId);
    return encoding;
  }

  async preload(imageIds: string[]): Promise<void> {
    // Background preload for project images
    for (const id of imageIds) {
      if (!await this.has(id)) {
        await this.get(id);  // Downloads and caches
      }
    }
  }
}
```

---

### Storage Options Comparison

| Option | Download 215MB | Cold Start | Cost (100 imgs) | Complexity |
|--------|----------------|------------|-----------------|------------|
| **A: Blob + SSD** | 3-4s (blob), 300ms (SSD) | 3-4s | ~$5/mo | Medium |
| **B: Files Premium** | 700ms | 700ms | ~$15/mo | Low |
| **C: Blob + Frontend** | 3-4s (once) | 50-200ms (cached) | ~$5/mo | Medium |

### Recommendation by Use Case

| Scenario | Recommended Option |
|----------|-------------------|
| Single worker, dev/test | A (Blob + SSD) |
| Production, consistent latency | B (Files Premium) |
| React frontend, offline support | C (Blob + Frontend) |
| Cost-sensitive, many images | A or C |

---

### Legacy Options (Reference)

| Option | Pros | Cons | Best For |
|--------|------|------|----------|
| **Local Safetensors** | Fast (76x vs pickle), simple | Single machine only | Development |
| **Redis + Safetensors** | Fast, shared | Memory-bound | Hot cache layer |
| **Azure NetApp Files** | <1ms latency, 4.5+ GB/s | Expensive (~$100+/mo) | Enterprise HPC |

### Why Safetensors?

Based on [Hugging Face safetensors](https://github.com/huggingface/safetensors):

- **76x faster** loading on CPU vs `torch.save`/pickle
- **Zero-copy** memory mapping - no RAM duplication
- **Safe** - no pickle code injection vulnerabilities
- **Lazy loading** - load only needed tensors
- **Framework agnostic** - works with PyTorch, ONNX, TensorFlow

### Storage Layout

```
encodings/
├── {image_id_1}.safetensors     # ~215MB per image
├── {image_id_1}.meta.json       # Metadata (sizes, timestamps)
├── {image_id_2}.safetensors
├── {image_id_2}.meta.json
└── ...
```

### Estimated Storage Requirements

| Images | Size (bfloat16) | Size (float32) |
|--------|-----------------|----------------|
| 50     | ~10.7 GB        | ~21.5 GB       |
| 100    | ~21.5 GB        | ~43 GB         |
| 500    | ~107 GB         | ~215 GB        |

---

## 2. Simplified State Caching

### Use Existing Processor API

Instead of dissecting internal structures, cache the full `state` dict:

```python
from sam3.model.sam3_image_processor import Sam3Processor

processor = Sam3Processor(model)

# Step 1: Encode image (expensive, ~2-3s)
state = processor.set_image(image)  # Returns dict with backbone_out

# Step 2: Run inference (fast, ~200-500ms) - can repeat many times
state = processor.set_text_prompt("person", state)
# or
state = processor.add_geometric_prompt(box, label, state)

# Results available in state
masks = state["masks"]
boxes = state["boxes"]
scores = state["scores"]
```

### What's in the State Dict

After `set_image()`:
```python
state = {
    "original_height": int,
    "original_width": int,
    "backbone_out": {
        "vision_features": Tensor,      # (1, 256, 36, 36)
        "backbone_fpn": [Tensor, ...],  # 4 scales
        "vision_pos_enc": [Tensor, ...],# 4 position encodings
        "sam2_backbone_out": {...},     # Optional SAM2 features
    }
}
```

After inference:
```python
state = {
    ...  # Above, plus:
    "geometric_prompt": Prompt,  # Accumulated prompts
    "masks": Tensor,             # Output masks
    "boxes": Tensor,             # Output boxes
    "scores": Tensor,            # Confidence scores
    "masks_logits": Tensor,      # Raw logits
}
```

### Serialization Strategy

Only serialize the `backbone_out` portion (the expensive encoding):

```python
import safetensors.torch as st

def save_encoding(state: dict, path: Path) -> None:
    """Save backbone encoding to disk."""
    backbone_out = state["backbone_out"]

    # Flatten nested structure for safetensors
    tensors = {}
    for i, fpn in enumerate(backbone_out["backbone_fpn"]):
        tensors[f"backbone_fpn_{i}"] = fpn.to(torch.bfloat16)
    for i, pos in enumerate(backbone_out["vision_pos_enc"]):
        tensors[f"vision_pos_enc_{i}"] = pos.to(torch.bfloat16)

    # Handle optional SAM2 backbone
    if backbone_out.get("sam2_backbone_out"):
        sam2 = backbone_out["sam2_backbone_out"]
        for i, fpn in enumerate(sam2["backbone_fpn"]):
            tensors[f"sam2_backbone_fpn_{i}"] = fpn.to(torch.bfloat16)
        for i, pos in enumerate(sam2["vision_pos_enc"]):
            tensors[f"sam2_vision_pos_enc_{i}"] = pos.to(torch.bfloat16)

    st.save_file(tensors, path)

    # Save metadata separately
    meta = {
        "original_height": state["original_height"],
        "original_width": state["original_width"],
        "created_at": datetime.now().isoformat(),
    }
    path.with_suffix(".meta.json").write_text(json.dumps(meta))


def load_encoding(path: Path, device: str = "cuda") -> dict:
    """Load backbone encoding from disk."""
    tensors = st.load_file(path, device=device)
    meta = json.loads(path.with_suffix(".meta.json").read_text())

    # Reconstruct backbone_out structure
    backbone_fpn = [tensors[f"backbone_fpn_{i}"] for i in range(4)]
    vision_pos_enc = [tensors[f"vision_pos_enc_{i}"] for i in range(4)]

    backbone_out = {
        "vision_features": backbone_fpn[-1],
        "backbone_fpn": backbone_fpn,
        "vision_pos_enc": vision_pos_enc,
    }

    # Reconstruct SAM2 if present
    if "sam2_backbone_fpn_0" in tensors:
        sam2_fpn = [tensors[f"sam2_backbone_fpn_{i}"] for i in range(4)]
        sam2_pos = [tensors[f"sam2_vision_pos_enc_{i}"] for i in range(4)]
        backbone_out["sam2_backbone_out"] = {
            "vision_features": sam2_fpn[-1],
            "backbone_fpn": sam2_fpn,
            "vision_pos_enc": sam2_pos,
        }

    return {
        "original_height": meta["original_height"],
        "original_width": meta["original_width"],
        "backbone_out": backbone_out,
    }
```

---

## 3. ONNX Export Strategy

### Available Tools

Based on community tools for SAM2:

| Tool | Link | Notes |
|------|------|-------|
| **samexporter** | [GitHub](https://github.com/vietanhdev/samexporter) | Supports SAM2, pip installable |
| **ONNX-SAM2-Segment-Anything** | [GitHub](https://github.com/ibaiGorordo/ONNX-SAM2-Segment-Anything) | Python inference examples |
| **SAM2ONNX** | [GitHub](https://github.com/DmitryYurov/SAM2ONNX) | Encoder + decoder split |
| **Pre-converted models** | [HuggingFace](https://huggingface.co/shubham0204/sam2-onnx-models) | Ready to use |

### The Encoder/Decoder Split Pattern

All ONNX export approaches split the model into two parts:

```
┌─────────────────────┐      ┌─────────────────────┐
│   Image Encoder     │      │      Decoder        │
│   (ViT Backbone)    │─────▶│  (Mask Generation)  │
│                     │      │                     │
│   ~2GB, slow        │      │   ~50MB, fast       │
│   Run once/image    │      │   Run per prompt    │
└─────────────────────┘      └─────────────────────┘
     encoder.onnx              decoder.onnx
```

### Export Script (Conceptual)

```python
# Using samexporter pattern
from samexporter import export_sam2

# Export encoder (image embedding)
export_sam2.export_encoder(
    model_path="sam2_hiera_large.pt",
    output_path="encoder.onnx",
    opset=17,
)

# Export decoder (mask generation)
export_sam2.export_decoder(
    model_path="sam2_hiera_large.pt",
    output_path="decoder.onnx",
    opset=17,
)
```

### SAM3 ONNX Export (Investigation Needed)

SAM3 builds on SAM2 but has additional components:
- Text encoder (for text prompts)
- Geometry encoder (for box/point prompts)
- More complex decoder

Options:
1. **Adapt samexporter** for SAM3 architecture
2. **Use torch.onnx.export** with careful tracing
3. **Export only decoder** - keep encoder on backend

### Browser Deployment Path

Based on [Medium tutorial](https://medium.com/@geronimo7/in-browser-image-segmentation-with-segment-anything-model-2-c72680170d92):

```
Phase 1: Backend Only
┌──────────────────────────────────────────┐
│           Backend (PyTorch)              │
│  Encoder + Decoder  ──▶  Masks           │
└──────────────────────────────────────────┘

Phase 2: Backend Encoder + Frontend Decoder
┌──────────────┐         ┌──────────────────┐
│   Backend    │ encode  │    Frontend      │
│  (PyTorch)   │────────▶│   (ONNX RT)      │
│   Encoder    │  ~215MB │    Decoder       │
└──────────────┘         └──────────────────┘

Phase 3: Full Frontend (Optional)
┌──────────────────────────────────────────┐
│        Frontend (ONNX RT + WebGPU)       │
│  Encoder (~2GB) + Decoder  ──▶  Masks    │
└──────────────────────────────────────────┘
```

---

## 4. Updated Service Design

### EncodingStore Interface

```python
from abc import ABC, abstractmethod
from pathlib import Path
from uuid import UUID

class EncodingStore(ABC):
    """Abstract interface for encoding persistence."""

    @abstractmethod
    def exists(self, image_id: UUID) -> bool:
        """Check if encoding exists."""
        pass

    @abstractmethod
    def save(self, image_id: UUID, state: dict) -> None:
        """Save backbone encoding."""
        pass

    @abstractmethod
    def load(self, image_id: UUID, device: str = "cuda") -> dict:
        """Load backbone encoding."""
        pass

    @abstractmethod
    def delete(self, image_id: UUID) -> None:
        """Delete encoding."""
        pass


class LocalEncodingStore(EncodingStore):
    """File-based encoding store using safetensors."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, image_id: UUID) -> Path:
        return self.base_dir / f"{image_id}.safetensors"

    def exists(self, image_id: UUID) -> bool:
        return self._path(image_id).exists()

    def save(self, image_id: UUID, state: dict) -> None:
        save_encoding(state, self._path(image_id))

    def load(self, image_id: UUID, device: str = "cuda") -> dict:
        return load_encoding(self._path(image_id), device)

    def delete(self, image_id: UUID) -> None:
        self._path(image_id).unlink(missing_ok=True)
        self._path(image_id).with_suffix(".meta.json").unlink(missing_ok=True)


class AzureBlobEncodingStore(EncodingStore):
    """Azure Blob Storage based encoding store."""

    def __init__(self, connection_string: str, container: str = "encodings"):
        from azure.storage.blob import ContainerClient
        self.client = ContainerClient.from_connection_string(
            connection_string, container
        )

    def save(self, image_id: UUID, state: dict) -> None:
        # Save to temp file, upload to blob
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            save_encoding(state, Path(f.name))
            self.client.upload_blob(f"{image_id}.safetensors", f)

    def load(self, image_id: UUID, device: str = "cuda") -> dict:
        # Download to temp file, load
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            blob = self.client.download_blob(f"{image_id}.safetensors")
            blob.readinto(f)
            return load_encoding(Path(f.name), device)
```

### Updated SAM3Service

```python
class SAM3Service:
    """Service with persistent encoding cache."""

    def __init__(
        self,
        encoding_store: EncodingStore | None = None,
        local_cache_size: int = 0,  # 0 = disabled
    ):
        self._model = None
        self._processor = None
        self._store = encoding_store or LocalEncodingStore(Path("./encodings"))

        # Optional in-memory LRU cache for hot encodings
        self._local_cache: OrderedDict[UUID, dict] = OrderedDict()
        self._local_cache_size = local_cache_size

    def encode_image(self, image_id: UUID, image: Image.Image) -> None:
        """Encode image and persist to store."""
        if self._store.exists(image_id):
            return  # Already encoded

        state = self._processor.set_image(image)
        self._store.save(image_id, state)

        # Optionally cache locally
        if self._local_cache_size > 0:
            self._cache_locally(image_id, state)

    def get_encoding(self, image_id: UUID) -> dict:
        """Get encoding from cache or store."""
        # Check local cache first
        if image_id in self._local_cache:
            self._local_cache.move_to_end(image_id)
            return self._local_cache[image_id]

        # Load from store
        state = self._store.load(image_id)

        # Cache locally if enabled
        if self._local_cache_size > 0:
            self._cache_locally(image_id, state)

        return state

    def infer(
        self,
        image_id: UUID,
        text_prompt: str | None = None,
        boxes: list[tuple[list[float], bool]] | None = None,
    ) -> dict:
        """Run inference using cached encoding."""
        state = self.get_encoding(image_id)

        # Reset prompts from any previous inference
        self._processor.reset_all_prompts(state)

        # Apply prompts
        if text_prompt:
            state = self._processor.set_text_prompt(text_prompt, state)

        if boxes:
            for box, is_positive in boxes:
                state = self._processor.add_geometric_prompt(box, is_positive, state)

        return {
            "masks": state.get("masks"),
            "boxes": state.get("boxes"),
            "scores": state.get("scores"),
        }

    def _cache_locally(self, image_id: UUID, state: dict) -> None:
        """Add to local LRU cache."""
        while len(self._local_cache) >= self._local_cache_size:
            self._local_cache.popitem(last=False)
        self._local_cache[image_id] = state
```

---

## 5. API Design

### REST Endpoints

```python
# Encode image (call when image selected for annotation)
POST /api/v1/images/{image_id}/encode
Response: {
    "status": "encoded",
    "encoding_id": "uuid",
    "size_mb": 215.0,
    "latency_ms": 2500
}

# Check if encoded
GET /api/v1/images/{image_id}/encoding
Response: {
    "exists": true,
    "created_at": "2025-01-15T10:00:00Z",
    "size_mb": 215.0
}

# Run inference with prompts
POST /api/v1/images/{image_id}/infer
Body: {
    "text_prompt": "person",  // optional
    "boxes": [                // optional
        {"coords": [0.1, 0.2, 0.3, 0.4], "positive": true}
    ],
    "confidence_threshold": 0.5
}
Response: {
    "masks": [...],  // Base64 encoded or URLs
    "boxes": [...],
    "scores": [...],
    "latency_ms": 250
}

# Batch encode (background job)
POST /api/v1/encodings/batch
Body: {"image_ids": ["uuid1", "uuid2", ...]}
Response: {"job_id": "uuid", "status": "started"}
```

### WebSocket for Interactive Mode

```python
# Connect to interactive session
WS /api/v1/images/{image_id}/interactive

# Client sends prompts
{
    "type": "add_box",
    "box": [0.1, 0.2, 0.3, 0.4],
    "positive": true
}

# Server responds with results
{
    "type": "inference_result",
    "masks": [...],
    "latency_ms": 200
}
```

---

## 6. Performance Expectations

| Scenario | Latency | Notes |
|----------|---------|-------|
| **Upload + Encode** | ~2-3s (background) | User doesn't wait |
| **First annotation (SSD hit)** | **500-800ms** | Encoding pre-cached |
| **First annotation (Blob miss)** | **4-5s** | Download + cache + infer |
| **Subsequent prompts (same image)** | **200-500ms** | Already in GPU memory |
| **Switch images (SSD hit)** | **500-800ms** | Load from local cache |
| **Switch images (Blob miss)** | **4-5s** | Cold load from blob |

### User Experience Timeline

```
Upload:     [======= 2-3s background =======]
                                              ✓ Ready to annotate

Annotate:   [== 500ms ==]  First prompt
            [= 200ms =]    Add box
            [= 200ms =]    Adjust box
            [= 200ms =]    Add another box
```

---

## 7. Encode-on-Upload Workflow

### Database Schema Addition

```sql
-- Add encoding status to images table
ALTER TABLE images ADD COLUMN encoding_status VARCHAR(20) DEFAULT 'pending';
-- Values: 'pending', 'processing', 'completed', 'failed'

ALTER TABLE images ADD COLUMN encoding_error TEXT;
ALTER TABLE images ADD COLUMN encoded_at TIMESTAMP;
```

### Upload Flow

```python
# In image upload endpoint
@router.post("/images")
async def upload_image(file: UploadFile, db: Session, background_tasks: BackgroundTasks):
    # 1. Save image to blob storage
    image = create_image_record(db, file)
    await save_to_blob(file, image.id)

    # 2. Queue encoding job (background)
    background_tasks.add_task(encode_image_task, image.id)

    return {"id": image.id, "encoding_status": "pending"}


async def encode_image_task(image_id: UUID):
    """Background task to encode image."""
    db = get_db_session()
    try:
        # Update status
        db.execute(
            update(Image)
            .where(Image.id == image_id)
            .values(encoding_status="processing")
        )
        db.commit()

        # Download image from blob
        image_data = await download_from_blob(image_id)
        pil_image = Image.open(io.BytesIO(image_data))

        # Encode with SAM3
        sam3_service = get_sam3_service()
        state = sam3_service.encode_image(pil_image)

        # Save encoding to blob
        await save_encoding_to_blob(image_id, state)

        # Update status
        db.execute(
            update(Image)
            .where(Image.id == image_id)
            .values(encoding_status="completed", encoded_at=datetime.utcnow())
        )
        db.commit()

    except Exception as e:
        db.execute(
            update(Image)
            .where(Image.id == image_id)
            .values(encoding_status="failed", encoding_error=str(e))
        )
        db.commit()
        raise
```

### API Endpoints for Encoding Status

```python
# Check encoding status
GET /api/v1/images/{image_id}
Response: {
    "id": "uuid",
    "filename": "photo.jpg",
    "encoding_status": "completed",  # pending, processing, completed, failed
    "encoded_at": "2025-01-15T10:00:00Z"
}

# Bulk status check
GET /api/v1/images?ids=uuid1,uuid2,uuid3
Response: {
    "images": [
        {"id": "uuid1", "encoding_status": "completed"},
        {"id": "uuid2", "encoding_status": "processing"},
        {"id": "uuid3", "encoding_status": "pending"}
    ]
}

# Retry failed encoding
POST /api/v1/images/{image_id}/encode
Response: {"status": "queued"}
```

### Frontend Considerations

```typescript
// Poll for encoding status after upload
const uploadImage = async (file: File) => {
  const response = await api.uploadImage(file);
  const imageId = response.id;

  // Poll until encoded
  while (true) {
    const status = await api.getImage(imageId);
    if (status.encoding_status === 'completed') {
      return imageId;
    }
    if (status.encoding_status === 'failed') {
      throw new Error(status.encoding_error);
    }
    await sleep(1000); // Poll every second
  }
};
```

---

## 8. Implementation Phases

### Phase 1: Core Encoding Infrastructure
1. Add `safetensors` dependency
2. Implement `save_encoding()` / `load_encoding()` functions
3. Implement `LocalEncodingStore` (SSD cache)
4. Implement `AzureBlobEncodingStore` (persistent)
5. Update `SAM3Service` with `encode_image()` and `infer_with_encoding()`

### Phase 2: Encode-on-Upload
1. Add `encoding_status` column to images table
2. Create background encoding task
3. Update upload endpoint to queue encoding
4. Add encoding status API endpoints
5. Add retry mechanism for failed encodings

### Phase 3: Two-Tier Cache
1. Implement `TieredEncodingStore` (SSD + Blob)
2. Add LRU eviction for local SSD cache
3. Add cache warming (preload active project images)

### Phase 4: Frontend Integration
1. Show encoding status in UI (spinner, checkmark)
2. Disable annotation until encoding complete
3. Update inference calls to use cached encodings

### Phase 5: Frontend Caching (Option C)
1. Implement encoding download endpoint (returns safetensors as binary)
2. Build IndexedDB cache wrapper in TypeScript
3. Add preload API for batch downloading
4. Implement cache eviction (LRU by last access)

### Phase 6: ONNX Frontend Inference
1. Investigate SAM3-specific ONNX export
2. Test decoder-only export
3. Prototype React + ONNX Runtime + WebGPU integration
4. Benchmark browser inference performance

---

## 9. Frontend Caching Details (Option C)

### Browser Storage Limits

| Browser | IndexedDB Limit | Notes |
|---------|-----------------|-------|
| Chrome | 60% of disk (up to 80GB typical) | Per-origin quota |
| Firefox | 50% of disk | Per-origin quota |
| Safari | 1GB default, can request more | User may be prompted |
| Edge | Same as Chrome | Chromium-based |

For 50 images × 215MB = ~10.7GB - well within Chrome/Firefox limits.

### IndexedDB Implementation

```typescript
import { openDB, DBSchema, IDBPDatabase } from 'idb';

interface EncodingDB extends DBSchema {
  encodings: {
    key: string;  // image_id
    value: {
      data: ArrayBuffer;
      imageId: string;
      createdAt: number;
      lastAccessed: number;
      sizeBytes: number;
    };
    indexes: { 'by-last-accessed': number };
  };
  metadata: {
    key: string;
    value: { totalSize: number; count: number };
  };
}

class FrontendEncodingCache {
  private db: IDBPDatabase<EncodingDB>;
  private maxSizeBytes: number;

  constructor(maxSizeGB: number = 10) {
    this.maxSizeBytes = maxSizeGB * 1024 * 1024 * 1024;
  }

  async init(): Promise<void> {
    this.db = await openDB<EncodingDB>('sam3-encodings', 1, {
      upgrade(db) {
        const store = db.createObjectStore('encodings', { keyPath: 'imageId' });
        store.createIndex('by-last-accessed', 'lastAccessed');
        db.createObjectStore('metadata');
      },
    });
  }

  async get(imageId: string): Promise<ArrayBuffer | null> {
    const entry = await this.db.get('encodings', imageId);
    if (entry) {
      // Update last accessed time
      entry.lastAccessed = Date.now();
      await this.db.put('encodings', entry);
      return entry.data;
    }
    return null;
  }

  async put(imageId: string, data: ArrayBuffer): Promise<void> {
    // Evict old entries if needed
    await this.ensureSpace(data.byteLength);

    await this.db.put('encodings', {
      imageId,
      data,
      createdAt: Date.now(),
      lastAccessed: Date.now(),
      sizeBytes: data.byteLength,
    });

    await this.updateMetadata();
  }

  private async ensureSpace(needed: number): Promise<void> {
    const meta = await this.db.get('metadata', 'stats') || { totalSize: 0, count: 0 };

    while (meta.totalSize + needed > this.maxSizeBytes) {
      // Evict least recently accessed
      const tx = this.db.transaction('encodings', 'readwrite');
      const index = tx.store.index('by-last-accessed');
      const cursor = await index.openCursor();

      if (cursor) {
        meta.totalSize -= cursor.value.sizeBytes;
        await cursor.delete();
      } else {
        break;
      }
    }
  }

  async preloadFromBackend(imageIds: string[]): Promise<void> {
    for (const imageId of imageIds) {
      if (await this.get(imageId)) continue;  // Already cached

      const response = await fetch(`/api/v1/images/${imageId}/encoding`);
      if (response.ok) {
        const data = await response.arrayBuffer();
        await this.put(imageId, data);
      }
    }
  }
}
```

### Backend Endpoint for Encoding Download

```python
@router.get("/images/{image_id}/encoding")
async def download_encoding(
    image_id: UUID,
    store: EncodingStore = Depends(get_encoding_store),
):
    """Download raw encoding for frontend caching."""
    if not store.exists(image_id):
        raise HTTPException(404, "Encoding not found")

    path = store.get_path(image_id)

    return FileResponse(
        path,
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "public, max-age=31536000",  # 1 year (immutable)
            "Content-Disposition": f"attachment; filename={image_id}.safetensors",
        }
    )
```

### Hybrid Mode: Backend Encode + Frontend Infer

This combines the best of both worlds:
- Backend does the heavy ViT encoding (requires GPU)
- Frontend caches encodings and runs lightweight decoder (ONNX)

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Hybrid Architecture                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Upload Flow (Backend):                                            │
│   Image → Backend GPU → Encode (2-3s) → Save to Blob                │
│                                                                     │
│   Annotation Flow (Frontend):                                       │
│   1. Check IndexedDB cache                                          │
│   2. If miss: Download from Blob (3-4s, one-time)                   │
│   3. Load into ONNX Runtime                                         │
│   4. Run decoder on each prompt (200-500ms)                         │
│                                                                     │
│   ┌──────────────┐         ┌──────────────┐        ┌─────────────┐  │
│   │   Backend    │ encode  │  Azure Blob  │  cache │  Frontend   │  │
│   │   (GPU)      │────────►│  (storage)   │───────►│  (ONNX)     │  │
│   │              │         │              │        │             │  │
│   │  ViT Encoder │         │  215MB/img   │        │  Decoder    │  │
│   └──────────────┘         └──────────────┘        └─────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Benefits:**
- No backend GPU needed during annotation (only for encoding)
- Scale annotation horizontally via frontend
- Offline annotation after initial cache
- Blob storage is cheap and sufficient

---

## Sources

- [Hugging Face safetensors](https://github.com/huggingface/safetensors) - Tensor serialization
- [samexporter](https://github.com/vietanhdev/samexporter) - SAM2 ONNX export
- [ONNX-SAM2-Segment-Anything](https://github.com/ibaiGorordo/ONNX-SAM2-Segment-Anything) - ONNX inference
- [In-Browser SAM2](https://medium.com/@geronimo7/in-browser-image-segmentation-with-segment-anything-model-2-c72680170d92) - Browser deployment
- [Pre-converted SAM2 ONNX](https://huggingface.co/shubham0204/sam2-onnx-models) - Ready models
