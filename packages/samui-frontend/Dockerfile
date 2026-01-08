FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy workspace files
COPY pyproject.toml ./
COPY packages/samui-frontend/pyproject.toml packages/samui-frontend/
COPY packages/samui-frontend/src packages/samui-frontend/src

# Create minimal backend pyproject.toml to satisfy workspace
RUN mkdir -p packages/samui-backend && \
    echo '[project]\nname = "samui-backend"\nversion = "0.1.0"\nrequires-python = ">=3.11"\ndependencies = []' > packages/samui-backend/pyproject.toml

# Install dependencies
RUN uv sync --package samui-frontend

EXPOSE 8501

CMD ["uv", "run", "--package", "samui-frontend", "streamlit", "run", "packages/samui-frontend/src/samui_frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
