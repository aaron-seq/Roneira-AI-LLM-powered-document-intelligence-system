# Container image for the Document Intelligence API.
#
# The previous file's `production` stage ran `app.main:app` and copied only
# `app/` and `config.py` — a tree the service does not use — while the
# `development` stage ran `backend.main:app`. Production and development were
# two different applications. Both stages now build and run the same code.

# ---------------------------------------------------------------- builder
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build toolchain is needed for wheels without a manylinux build; it stays in
# this stage and never reaches the runtime image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------- base
# Shared runtime layer: interpreter, OS packages, user, directories.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH"

# curl for the healthcheck; libmagic for content-type sniffing on upload;
# tesseract for OCR — it is a system package, so without it scanned documents
# would be refused inside the container while working on the host.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libmagic1 tesseract-ocr \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN useradd --create-home --shell /bin/bash --uid 1000 app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Writable paths for uploads, the SQLite database and the vector store.
RUN mkdir -p /app/uploads /app/processed /app/chroma_db /app/logs \
    && chown -R app:app /app /opt/venv

# ------------------------------------------------------------- production
FROM base AS production

COPY --chown=app:app backend/ ./backend/
COPY --chown=app:app gunicorn.conf.py ./

USER app

EXPOSE 8000

# Liveness must not depend on the database: a transient DB outage should not
# make the orchestrator kill an otherwise healthy container.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health/live || exit 1

# Uvicorn workers under Gunicorn: Gunicorn supervises and recycles processes,
# uvicorn provides the ASGI event loop.
CMD ["gunicorn", "backend.main:app", "--config", "gunicorn.conf.py"]

# ------------------------------------------------------------ development
FROM base AS development

# Source is bind-mounted by docker-compose, so nothing is copied here.
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health/live || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
