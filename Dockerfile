# HEAN API - Docker Image (uv workspace)

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install external Python dependencies (cached layer)
RUN pip install --no-cache-dir \
    pydantic>=2.0.0 \
    pydantic-settings>=2.0.0 \
    aiohttp>=3.9.0 \
    websockets>=12.0 \
    httpx>=0.25.0 \
    redis>=5.0.0 \
    orjson>=3.9.0 \
    python-dotenv>=1.0.0 \
    numpy>=1.24.0 \
    networkx>=3.2.0 \
    psutil>=5.9.0 \
    fastapi>=0.104.0 \
    "uvicorn[standard]>=0.24.0" \
    slowapi>=0.1.9 \
    prometheus-client>=0.19.0 \
    duckdb>=1.0.0 \
    polars>=0.20.0 \
    pyarrow>=15.0.0 \
    python-socketio>=5.10.0 \
    openai>=1.0.0 \
    anthropic>=0.30.0

# Copy workspace packages
COPY packages/ ./packages/

# Create data directories
RUN mkdir -p /app/data /app/logs

# PYTHONPATH: all workspace package src dirs for namespace resolution
ENV PYTHONPATH="/app/packages/hean-core/src:/app/packages/hean-exchange/src:/app/packages/hean-portfolio/src:/app/packages/hean-risk/src:/app/packages/hean-execution/src:/app/packages/hean-strategies/src:/app/packages/hean-physics/src:/app/packages/hean-intelligence/src:/app/packages/hean-observability/src:/app/packages/hean-symbiont/src:/app/packages/hean-api/src:/app/packages/hean-app/src"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# CRITICAL: workers=1 — TradingSystem uses in-process shared state (EventBus, accounting,
# killswitch). Multiple workers would create independent TradingSystem instances (split-brain):
# API queries worker-1 while trading happens in worker-2, showing stale positions/equity.
# For horizontal scaling, deploy multiple containers (not multiple workers per container).
CMD ["uvicorn", "hean.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
