FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# Install git (needed for sam3 from GitHub)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy backend package files (build context is project root)
COPY packages/samui-backend/pyproject.toml ./pyproject.toml
COPY packages/samui-backend/src ./src

# Install dependencies into system Python (reuses torch from base image)
RUN uv pip install --system -e .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "samui_backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
