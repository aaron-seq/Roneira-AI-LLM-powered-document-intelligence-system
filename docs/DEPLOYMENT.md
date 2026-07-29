# Deployment

Getting this into production without the surprises.

---

## Pre-flight checklist

Work through this before exposing the service. Items marked **required** will
prevent startup if missed — that is deliberate.

- [ ] **Required.** `SECRET_KEY` set to ≥32 random characters
      (`python -c "import secrets; print(secrets.token_urlsafe(48))"`)
- [ ] **Required.** `ENVIRONMENT=production` and `DEBUG=false`
- [ ] **Required.** `ALLOWED_ORIGINS` and `ALLOWED_HOSTS` set to real domains
      (not `*`)
- [ ] **Required.** `REQUIRE_AUTHENTICATION=true`
- [ ] Replace the built-in demo accounts — see [SECURITY.md](SECURITY.md)
- [ ] `DATABASE_URL` pointing at PostgreSQL if you run more than one worker
- [ ] `REQUIRE_REAL_EMBEDDINGS=true` so a missing model fails loudly rather
      than silently degrading every search to keyword matching
- [ ] TLS terminated in front of the service
- [ ] `UPLOAD_DIRECTORY` and `VECTOR_STORE_PATH` on persistent, backed-up
      storage
- [ ] Prometheus scraping `/api/metrics`
- [ ] A scheduled call to source-retention purging (see below)

With `ENVIRONMENT=production`, the settings validator refuses to start on a
placeholder key, a short key, disabled auth, wildcard CORS/hosts, or debug
mode. Read the startup error — it names every problem at once.

---

## Health endpoints

Wire these to the right things; they are not interchangeable.

| Endpoint | Use for | Behaviour |
|---|---|---|
| `/api/health/live` | Liveness probe | 200 while the process runs. **Does not touch the database** — a transient DB outage must not trigger a restart loop. |
| `/api/health/ready` | Readiness probe | 503 when the database is unreachable, so traffic stops arriving. |
| `/api/health` | Dashboards and alerting | Per-component detail; `degraded` when serving with reduced capability. |

`/api/health` reporting `degraded` is a real signal, not noise. It means the
service is answering requests while something is wrong — usually a missing
embedding model (search is keyword-only) or an unreachable LLM (no summaries).
Both are invisible to a plain up/down check.

---

## Scaling

**The default configuration is single-node.** Two constraints:

1. **SQLite** — one writer. Move to PostgreSQL before adding workers.
2. **ChromaDB persists to a local directory.** Multiple processes writing the
   same directory will corrupt it.

For a single instance:

```bash
WORKERS=1 gunicorn backend.main:app --config gunicorn.conf.py
```

To scale horizontally you need, in order:

1. PostgreSQL (`DATABASE_URL=postgresql+asyncpg://…`).
2. A shared vector store — ChromaDB in server mode, or Qdrant/Weaviate/pgvector.
   This requires a change to `VectorStoreService`.
3. Redis-backed conversation memory (currently in-process, so chat history is
   lost on restart and not shared between workers).
4. Redis-backed rate limiting (currently per-process, so N replicas give N ×
   the configured limit).
5. Shared object storage for uploads (S3 or similar) instead of a local
   directory.

Steps 3–5 are not implemented. Until they are, run one instance and scale
vertically — an honest single node beats a cluster with a corrupted index.

### Sizing

Document processing and LLM inference happen **inside the worker**, so a slow
model holds a worker for the duration. `WORKER_TIMEOUT` defaults to 180s for
that reason (Gunicorn's own default of 30s would kill workers mid-inference).

Rough starting point: 2 vCPU / 4GB RAM without a local embedding model,
4 vCPU / 8GB with one. The `all-MiniLM-L6-v2` model is roughly 90MB on disk and
several hundred MB resident.

---

## Docker

```bash
docker build --target production -t roneira:latest .

docker run -d -p 8000:8000 \
  -e ENVIRONMENT=production \
  -e DEBUG=false \
  -e SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')" \
  -e DATABASE_URL="postgresql+asyncpg://user:pass@db:5432/roneira" \
  -e ALLOWED_ORIGINS="https://app.example.com" \
  -e ALLOWED_HOSTS="api.example.com" \
  -e REQUIRE_REAL_EMBEDDINGS=true \
  -v roneira-uploads:/app/uploads \
  -v roneira-vectors:/app/chroma_db \
  roneira:latest
```

The image runs as a non-root user and its healthcheck uses `/api/health/live`.
**Mount volumes for `/app/uploads` and `/app/chroma_db`** — without them,
uploaded documents and the entire search index vanish when the container is
replaced.

Both the `production` and `development` stages build and run the same
application. They used to run different ones: `production` served `app.main:app`
and did not copy `backend/` at all.

---

## Platform notes

`deployment/` holds starting configurations for Railway, Render and Vercel.
Two things to know before using them:

- They were written for the earlier `app.main:app` entrypoint. The command is
  now `gunicorn backend.main:app --config gunicorn.conf.py`.
- **Vercel's serverless model does not fit this workload.** Document processing
  runs as a background task after the response is sent, and serverless
  functions are frozen once they respond. Uploads would be accepted and never
  processed. Use a platform with long-running processes and persistent disk.

Any platform you choose needs: persistent volumes, an environment where the
process outlives the request, and outbound access to Ollama if you want
generated answers.

---

## Observability

```bash
docker compose --profile observability up
```

Brings up Prometheus (`:9090`) scraping `/api/metrics`, and Grafana (`:3001`)
with the datasource and dashboard provisioned.

The metrics worth alerting on — all defined in
`deployment/prometheus/rules/alerts.yml`:

| Signal | Why it matters |
|---|---|
| `roneira_embedding_backend_real == 0` | Running keyword-only. Every search is worse and nobody is told. |
| `roneira_retrieval_queries_total{outcome="empty"}` ratio | Users are asking questions the index cannot answer. |
| `roneira_documents_processed_total{status="error"}` | Documents are failing to process. |
| `roneira_http_requests_total{status=~"5.."}` ratio | Ordinary error-rate alerting. |
| `roneira_http_request_latency_seconds` p95 | Latency regression. |

The first two are the ones a generic monitoring setup would miss entirely: the
service stays "up" and returns 200s while giving bad answers.

Logs are JSON in production (text locally) and carry `correlation_id`, which
matches the `X-Correlation-ID` response header. When a user reports an error,
ask for that ID.

---

## Backups

Three things must be backed up together, or a restore will be inconsistent:

1. **The database** — document records, chunk citation metadata, feedback.
2. **`UPLOAD_DIRECTORY`** — the original files.
3. **`VECTOR_STORE_PATH`** — the search index.

The vector store is the least critical: it is derived data and can be rebuilt
from the extracted text held in the database. There is no rebuild command yet;
re-indexing means calling `POST /api/documents/{id}/index` per document.

### Retention

`SOURCE_RETENTION_DAYS` bounds how long original files are kept, but **nothing
calls the purge automatically**. Schedule it:

```bash
python -c "
import asyncio
from backend.api.dependencies import initialize_services, cleanup_services

async def main():
    container = await initialize_services()
    removed = await container.document_processor.purge_expired_sources()
    print(f'removed {removed} expired source files')

asyncio.run(main())
"
```

---

## Database migrations

Alembic is in `requirements.txt` but no migration chain exists yet; the schema
is created with `Base.metadata.create_all()` at startup. That is fine for a
first deployment and **not** fine for the second: `create_all` adds missing
tables but never alters existing ones.

Before your first schema change, initialise Alembic and generate a baseline
against the current models. Until then, treat schema changes as requiring a
manual migration.
