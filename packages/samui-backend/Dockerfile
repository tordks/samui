FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy workspace files
COPY pyproject.toml ./
COPY packages/samui-backend/pyproject.toml packages/samui-backend/
COPY packages/samui-backend/src packages/samui-backend/src

# Create minimal frontend pyproject.toml to satisfy workspace
RUN mkdir -p packages/samui-frontend && \
    echo '[project]\nname = "samui-frontend"\nversion = "0.1.0"\nrequires-python = ">=3.11"\ndependencies = []' > packages/samui-frontend/pyproject.toml

# Install dependencies
RUN uv sync --package samui-backend

EXPOSE 8000

CMD ["uv", "run", "--package", "samui-backend", "uvicorn", "samui_backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
