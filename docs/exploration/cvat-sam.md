# CVAT SAM Plugin Architecture Analysis

Summary from exploration with Claude.

Research into how CVAT (Computer Vision Annotation Tool) implements the Segment Anything Model with a backend/frontend split.

## Overview

CVAT implements SAM as a **Nuclio serverless function** on the backend with a **frontend plugin** that handles decoding:

- **Backend (Serverless/Nuclio)**: Runs image encoding using PyTorch
- **Frontend (TypeScript/React)**: Decodes embeddings client-side using onnxruntime-web
- **Communication**: Embeddings transferred via base64-encoded arrays in JSON responses

---

## Architecture

### Two-Phase Split

```
Backend (Nuclio Serverless)          Frontend (React + Web Worker)
┌─────────────────────────┐          ┌─────────────────────────┐
│  Image Encoder (ViT-H)  │          │  Mask Decoder (ONNX)    │
│                         │  base64  │                         │
│  image → embeddings     │ ───────► │  embeddings + clicks    │
│  [1, 256, 64, 64]       │  ~4MB    │  → masks                │
│                         │          │                         │
│  Runs ONCE per image    │          │  Runs per USER CLICK    │
└─────────────────────────┘          └─────────────────────────┘
```

### Phase 1: Backend Image Encoding (runs once per image)

1. Backend serverless function receives image data (base64-encoded)
2. Runs SAM's image encoder (ViT-H backbone)
3. Generates embeddings with shape `[1, 256, 64, 64]` (batch, channels, height, width)
4. Embeddings represent 16x downscaling of original image
5. Returns embeddings to frontend as base64-encoded arrays in JSON

### Phase 2: Frontend Mask Decoding (runs per user interaction)

1. Frontend Web Worker loads decoder ONNX model (using onnxruntime-web)
2. Receives user clicks/boxes (scaled to model resolution)
3. Decodes embeddings + prompts into segmentation masks
4. Returns masks to annotation canvas
5. Uses LRU cache to store embeddings and low-res masks

---

## Embedding Transfer Format

### Protocol

- Embeddings encoded as base64 strings in JSON payloads
- Structure: `{ "embeddings": "<base64_string>", "low_res_mask": "<base64_string>" }`
- LRU cache on frontend prevents re-downloading embeddings for same frame
- Embeddings persist in memory during annotation session

### Embedding Dimensions

| Tensor | Shape | Type | Size |
|--------|-------|------|------|
| Image embeddings | `[1, 256, 64, 64]` | float32 | ~4MB |
| Low-res masks | `[1, 256, 256]` | float32 | ~256KB |

---

## Frontend Implementation

### Architecture

```
Frontend Architecture:
├── SAMPlugin (TypeScript interface)
│   ├── Enter Hook: Fetch/cache embeddings from backend
│   ├── Leave Hook: Decode embeddings on exit
│   └── Click Handler: Send prompts to worker
├── inference.worker.ts (Web Worker)
│   ├── Loads decoder.onnx via onnxruntime-web
│   ├── Processes clicks (point type classification: 0-3)
│   └── Returns mask predictions
└── LRU Cache (dual cache)
    ├── Embeddings cache [256, 64, 64]
    └── Masks cache [1, 256, 256]
```

### ONNX Export Requirements

- Image encoder and mask decoder must be separately exported to ONNX
- Only compatible with specific model weights (`sam_vit_h_4b8939.pth`)
- Other variants require re-exporting ONNX decoder with custom modifications

---

## SAM2 Challenges (PR #8243)

CVAT attempted SAM2 integration but faced architectural challenges:

### Problem: Stateful Memory Bank

SAM2 adds stateful memory bank for video tracking:
- Memory encoder creates FIFO queue of recent frame embeddings
- Memory attention module conditions current frame on past frames
- Stateful architecture doesn't fit stateless serverless model

### Resolution Options Considered

1. **Sticky sessions + GPU memory persistence** - not cloud-scalable
2. **Store entire memory bank as session state** - large transfer overhead
3. **Client-side memory bank** - complex onnxruntime-web implementation

### CVAT Decision

Kept SAM2 for paid enterprise/SaaS only due to these technical blockers.

---

## Why Hybrid Architecture?

### Comparison

| Aspect | Backend Only | Frontend Only | Hybrid (CVAT) |
|--------|-------------|--------------|---------------|
| GPU Cost | Expensive (per click) | Scales to users | Scales (encode once) |
| Latency | Network round trip | Instant | Good (cached embeddings) |
| Bandwidth | High (repeat requests) | Low | Moderate (one-time) |
| Deployment | Centralized | User devices | Flexible |
| Scalability | GPU bottleneck | Unlimited clients | GPU + client scaling |

### Key Insight (Issue #6049)

> "On large images, SAM latency per query adds up... database scaling is much easier (and cheaper) than scaling GPU hardware, particularly since embedding generation is computationally idempotent."

### Cost Model

- Without caching: O(users × clicks) GPU calls
- With embedding cache: O(unique images) GPU calls
- Frontend decoder: O(clicks) but runs on user's device (free)

---

## Why NOT Frontend-Only?

Despite onnxruntime-web's capabilities, CVAT keeps encoding backend-side:

### Browser Limitations

- Memory constraints with large images (HD/4K)
- JavaScript performance insufficient for ViT backbone
- Network costs shipping large model ONNX files to every user

### Operational Benefits

- Backend control over model versions and updates
- Easier A/B testing different SAM variants
- GPU resource pooling across users
- Security (model weights stay server-side)

### Cost Efficiency

- One GPU instance serves many users
- Embedding caching at database level (multi-session reuse)
- Bandwidth savings vs repeated full inference

---

## Performance

### Latency Breakdown (from PR #8243)

| Operation | Time |
|-----------|------|
| Backend inference (full) | 300-400ms |
| Network/integration overhead | 4.5-5s total |
| Frontend decoder (ONNX) | ~100ms |

### Optimization Techniques

- Image encoder runs only once per frame (idempotent)
- Results cached in memory (LRU) and optionally in database
- Decoder runs client-side (zero latency for repeated clicks)
- Batching enables scaling to multiple concurrent users

---

## Lessons for SAM3 Implementation

### 1. Use Hybrid Architecture

- Run image encoder once on backend (GPU-intensive)
- Cache embeddings in storage with image-level keys
- Run decoder on frontend or keep stateless on backend

### 2. Embedding Storage Strategy

- Create table/storage with composite key `(image_id, model_version)`
- Store embeddings as binary or compressed arrays
- Implement TTL-based eviction if storage becomes constraint

### 3. Embedding Size Comparison

| Model | Embedding Shape | Size |
|-------|-----------------|------|
| SAM1 (CVAT) | `[1, 256, 64, 64]` | ~4MB |
| SAM3 (full FPN) | Multi-scale pyramid | ~215MB |

**Question**: Can SAM3's decoder work with just the final layer (~4MB) instead of full FPN?

### 4. State Management for Find-All Mode

- Don't cache entire model state like SAM2 memory banks
- Cache per-image embeddings, run find-all batch inference once
- Store discovered annotations (source=MODEL) in database

### 5. ONNX Export Considerations

- Export encoder and decoder separately to ONNX
- Document specific model weight compatibility
- Version embeddings by model checkpoint
- Plan for future model variants requiring re-export

### 6. Video/Sequential Processing (Future)

- SAM3 has native video tracking unlike SAM1
- Don't replicate SAM2's memory bank in stateless architecture
- Process frames sequentially, cache inter-frame embeddings
- Return complete tracking sequences in single response

---

## CVAT-Inspired Architecture for SAMUI

```
┌──────────────────────────────────────────────────────────────────┐
│                    Proposed Architecture                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Backend (encode on upload):                                     │
│  ┌────────────────┐      ┌────────────────┐                      │
│  │ Image Upload   │ ───► │ ViT Encoder    │ ───► Blob Storage    │
│  └────────────────┘      │ (2-3s, GPU)    │      (embeddings)    │
│                          └────────────────┘                      │
│                                                                  │
│  Frontend (annotation):                                          │
│  ┌────────────────┐      ┌────────────────┐      ┌────────────┐  │
│  │ Select Image   │ ───► │ Fetch Embed    │ ───► │ IndexedDB  │  │
│  │                │      │ (3-4s once)    │      │ Cache      │  │
│  └────────────────┘      └────────────────┘      └────────────┘  │
│                                    │                             │
│                                    ▼                             │
│  ┌────────────────┐      ┌────────────────┐                      │
│  │ User Clicks    │ ───► │ ONNX Decoder   │ ───► Mask Display    │
│  │ Box/Point      │      │ (Web Worker)   │                      │
│  └────────────────┘      │ (~100ms)       │                      │
│                          └────────────────┘                      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Sources

- [CVAT SAM Plugin Implementation](https://github.com/cvat-ai/cvat/blob/develop/cvat-ui/plugins/sam/src/ts/index.tsx)
- [Better Integration for Neural Embedding based workflows · Issue #6049](https://github.com/cvat-ai/cvat/issues/6049)
- [Introduce Segment Anything 2 · PR #8243](https://github.com/cvat-ai/cvat/pull/8243)
- [Segment Anything Model 3 in CVAT, Part 1 | CVAT Blog](https://www.cvat.ai/resources/blog/sam-3-image-segmentation)
- [Meta's Segment Anything Model is now available in CVAT | CVAT Blog](https://www.cvat.ai/resources/blog/segment-anything-model-in-cvat)
- [Image Segmentation in the Browser with SAM2 | Medium](https://medium.com/@geronimo7/in-browser-image-segmentation-with-segment-anything-model-2-c72680170d92)
- [Export SAM to ONNX: the missing parts | DEV Community](https://dev.to/andreygermanov/export-segment-anything-neural-network-to-onnx-the-missing-parts-43c8)
