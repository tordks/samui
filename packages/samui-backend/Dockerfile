FROM python:3.11-slim

WORKDIR /app

# Install git (needed for sam3 from GitHub)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy backend package files (build context is project root)
COPY packages/samui-backend/pyproject.toml ./pyproject.toml
COPY packages/samui-backend/src ./src

# Install dependencies
RUN uv sync

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "samui_backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
