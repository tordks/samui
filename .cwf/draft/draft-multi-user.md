# Multi-User Architecture Analysis (Draft)

This document analyzes what changes would be required to support multiple users in the SAM3 WebUI application.

## Current Single-User Assumptions

| Aspect | Current Design | Implication |
|--------|----------------|-------------|
| Job queue | Global FIFO, one runs at a time | All users share one queue |
| Data access | No filtering by user | Anyone sees everything |
| BackgroundTasks | Process-local execution | Single API instance |
| Race conditions | "Acceptable" | Would cause data corruption |
| GPU | Single model instance | Serialized inference |

---

## Changes Required for Multi-User

### 1. User Model and Authentication

```python
# New model
class User(Base):
    id: UUID
    email: str  # unique
    created_at: datetime

# Add user_id FK to existing models
class Image(Base):
    user_id: UUID  # FK to User, required

class ProcessingJob(Base):
    user_id: UUID  # FK to User, required
```

**Auth options:**
- Azure AD / OAuth2 for enterprise
- Simple API key for internal tools
- Session-based for web UI

### 2. Data Isolation (Row-Level Security)

Every query must filter by user:

```python
# Current (single-user)
def get_images(db: Session) -> list[Image]:
    return db.query(Image).all()

# Multi-user
def get_images(db: Session, user_id: UUID) -> list[Image]:
    return db.query(Image).filter(Image.user_id == user_id).all()
```

**Options:**
- **Application-level filtering:** Add `user_id` parameter to all queries (simple, error-prone)
- **PostgreSQL RLS:** Row-level security policies enforce isolation at DB level (more secure)

```sql
-- PostgreSQL RLS example
ALTER TABLE images ENABLE ROW LEVEL SECURITY;
CREATE POLICY user_isolation ON images
    USING (user_id = current_setting('app.current_user_id')::uuid);
```

### 3. Job Queue Strategy

**Option A: Per-User Queue (Recommended)**

```python
class ProcessingJob(Base):
    user_id: UUID
    # Each user has independent queue
    # User A's jobs don't block User B

def start_job_if_none_running(db, user_id, job_id):
    running = db.query(ProcessingJob).filter(
        ProcessingJob.user_id == user_id,
        ProcessingJob.status == JobStatus.RUNNING
    ).first()
    if not running:
        # Start this user's job
```

- Pro: Fair, users don't block each other
- Con: Concurrent GPU access needs management

**Option B: Global Queue with Priority**

```python
class ProcessingJob(Base):
    user_id: UUID
    priority: int  # Higher = sooner

# Round-robin or fair scheduling across users
```

- Pro: Simple GPU management (one at a time)
- Con: Heavy user can starve others

### 4. Background Job Execution

BackgroundTasks is **not suitable** for multi-user:
- Tied to single process
- No persistence across restarts
- No distributed execution

**Required: External Job Queue**

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   API       │────▶│  Queue      │────▶│   Worker    │
│  (stateless)│     │ (Redis/SQS) │     │  (GPU)      │
└─────────────┘     └─────────────┘     └─────────────┘
```

Options:
- **Celery + Redis:** Battle-tested, Python-native
- **Azure Queue Storage:** Managed, integrates with Azure
- **AWS SQS + Lambda:** Serverless option
- **Database polling:** Simple but less scalable (current plan's "future work")

### 5. Concurrency and Race Conditions

**Current acceptable races become bugs:**

| Race Condition | Single-User | Multi-User |
|----------------|-------------|------------|
| Concurrent job start | Unlikely | Common |
| State check during update | Rare | Frequent |
| Duplicate inserts | Manual use | API clients |

**Solutions:**

```python
# Use database transactions with SELECT FOR UPDATE
def start_next_job(db, user_id):
    with db.begin():
        job = db.query(ProcessingJob).filter(
            ProcessingJob.user_id == user_id,
            ProcessingJob.status == JobStatus.QUEUED
        ).with_for_update(skip_locked=True).first()

        if job:
            job.status = JobStatus.RUNNING
            return job.id
```

### 6. Resource Management

**GPU Sharing:**
- Multiple workers with one GPU each, or
- Single worker with request queuing, or
- GPU scheduling (NVIDIA MPS/MIG)

**Quotas:**

```python
class User(Base):
    max_concurrent_jobs: int = 1
    max_images: int = 1000
    max_storage_mb: int = 5000
```

**Storage Isolation:**

```python
# Current: flat blob structure
images/{image_id}.png

# Multi-user: user-prefixed
users/{user_id}/images/{image_id}.png
```

---

## Database Schema Changes Summary

```python
# New table
class User(Base):
    id: UUID
    email: str
    created_at: datetime
    max_concurrent_jobs: int = 1

# Modified tables - add user_id FK
class Image(Base):
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"))

class Annotation(Base):
    # Inherits user scope via image.user_id (no direct FK needed)

class ProcessingJob(Base):
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"))

class ProcessingResult(Base):
    # Inherits user scope via job.user_id (no direct FK needed)
```

**Indexes for multi-user queries:**

```python
# Add composite indexes for filtered queries
Index("ix_images_user_id", Image.user_id)
Index("ix_jobs_user_status", ProcessingJob.user_id, ProcessingJob.status)
```

---

## Migration Path

| Phase | Change | Effort |
|-------|--------|--------|
| 1 | Add User model, user_id FKs (nullable initially) | Low |
| 2 | Add auth middleware, populate user_id | Medium |
| 3 | Make user_id required, add query filters | Medium |
| 4 | Replace BackgroundTasks with queue (Celery/Azure Queue) | High |
| 5 | Add RLS policies (optional, defense in depth) | Low |
| 6 | Add quotas and resource management | Medium |

---

## Relationship to Current Plan

The current job-based processing plan (03-job-processing) provides a good foundation for multi-user:

- **ProcessingJob table:** Already has the right structure, just needs `user_id`
- **Job processor service:** Can be extracted to separate worker process
- **Database-backed queue:** Natural migration path to external queue
- **Annotation ID set comparison:** Works unchanged per-user

The main architectural change is extracting the worker to a separate service that polls the database or message queue, which is already documented in the plan's "Future Work" section.
