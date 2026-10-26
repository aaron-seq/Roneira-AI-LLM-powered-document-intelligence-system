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
| `/api/health/live` | Liveness probe | 200 as soon as the process is up, ~1.5s from start. **Touches neither the database nor the embedding model** — a transient DB outage must not trigger a restart loop, and a model download must not look like a dead container. |
| `/api/health/ready` | Readiness probe, and anything that routes traffic | 503 when the database is unreachable **or the embedding model is still loading**, so traffic stops arriving. |
| `/api/health` | Dashboards and alerting | Per-component detail; `starting`, `degraded` or `unhealthy`. |

**Route traffic on readiness, not liveness.** The embedding model loads in a
background task so the server can start answering immediately, and a search
arriving before it finishes waits for it — a slow request, not a failed one,
but not something to send traffic into while another replica is ready. The
platform configs in `deployment/` use `/api/health/ready` for this reason;
Docker's own `HEALTHCHECK` uses liveness, because there its job is to decide
whether to restart the container.

Readiness resolves either way. A model that cannot be downloaded at all leaves
the service keyword-only and **ready** rather than failing readiness forever,
so a deployment with no model access still passes its health check.

`/api/health` reporting `degraded` is a real signal, not noise. It means the
service is answering requests while something is wrong — usually a missing
embedding model (search is keyword-only) or an unreachable LLM (no summaries).
Both are invisible to a plain up/down check. `starting` is distinct and not a
cause for alarm: nothing is missing, the model is still arriving.

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

`deployment/` holds starting configurations for Railway and Render. Both run
`uvicorn backend.main:app`, health-check `/api/health/ready`, and are pinned to
one instance because the vector store and uploaded files live on the
instance's own disk — a second replica would answer from an empty index.

**There is no Vercel config, deliberately.** One used to be here and could
never have worked: it pointed at the deleted `app/main.py` and passed Azure
and Redis variables the application does not read. Beyond that the platform is
the wrong shape for this workload — document processing runs as a background
task *after* the response is sent, and a serverless function is frozen once it
responds, so uploads would be accepted and never processed. There is also no
persistent disk for the vector store, and the sentence-transformers stack
alone exceeds the bundle limit. Removing it was the honest fix; a config that
looks supported but cannot run is worse than none.

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

The application logs to stdout. Use the Compose service name to inspect it:

```bash
docker compose logs --tail 100 --timestamps api
docker compose logs --follow api
```

For production, use `docker compose -f docker-compose.production.yml logs api`.
The Compose files do not configure log rotation. Set a bounded Docker logging
policy before long-running use; for example, the
[json-file driver's `max-size` and `max-file` options](https://docs.docker.com/engine/logging/drivers/json-file/).
A host file mount is not required for stdout
collection. Configure any external log collector for the data it will receive.

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

The schema is managed by Alembic, in `migrations/`. By default the application
applies any pending revisions during startup, so a single-container deployment
needs no extra step.

**For anything with more than one replica, turn that off:**

```bash
DB_AUTO_MIGRATE=false
```

and run migrations as their own deploy step, before the new version starts:

```bash
alembic upgrade head
```

A schema change is then a visible, ordered action with a log and an exit code,
rather than something applied by whichever replica happens to boot first. With
`DB_AUTO_MIGRATE=false`, startup checks that the tables exist and refuses to
start if they do not — naming the missing ones — instead of failing later
inside a request.

### Upgrading from a release before migrations existed

Nothing to do. The schema used to be created with `Base.metadata.create_all()`,
which leaves a database with all the tables and no `alembic_version` row.
Startup detects exactly that and stamps it at the baseline revision rather than
replaying the baseline against tables that already exist. Your data is
untouched.

### Reviewing a migration before it runs

```bash
alembic upgrade head --sql    # print the SQL instead of executing it
alembic current               # which revision this database is at
alembic history --verbose     # what exists
alembic downgrade -1          # step back one revision
```

`migrations/README.md` covers writing one. CI runs `alembic check`, which fails
when the models describe a schema the migrations do not produce — that is what
stops a model change from being merged without its migration.
