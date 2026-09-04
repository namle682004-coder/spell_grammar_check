FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Dependency layer
COPY pyproject.toml uv.lock README.md ./

RUN pip install --no-cache-dir --upgrade pip setuptools wheel uv

RUN uv sync --no-dev

# Create non-root user for security compliance
RUN useradd -u 1000 -m appuser && chown -R appuser:appuser /app

# Source code
COPY --chown=appuser:appuser src ./src

USER appuser

ENV PYTHONUNBUFFERED=1
ENV API_PORT=8080

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]