FROM python:3.12-slim

LABEL maintainer="VLM-RL Project"
LABEL description="Multi-agent active perception grasping system"

WORKDIR /app

# Copy all project files
COPY . /app

# Install uv package manager
RUN pip install --no-cache-dir uv

# Set uv build env vars (aligned with a2a-samples Containerfile)
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies (with cache mount for faster rebuilds)
RUN --mount=type=cache,target=/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Install the project itself
RUN --mount=type=cache,target=/.cache/uv \
    uv sync --frozen --no-dev

# All agent nodes run within this single container — no EXPOSE needed
# (Agent nodes do not open HTTP ports; A2A calls go OUT to RTX 3090)

# Default: run in mock mode
CMD ["uv", "run", "python", "main.py", "--mock"]
