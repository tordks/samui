FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy frontend package files (build context is project root)
COPY packages/samui-frontend/pyproject.toml ./pyproject.toml
COPY packages/samui-frontend/src ./src

# Install dependencies
RUN uv sync

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "src/samui_frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
