# Architecture

How the system is put together, what happens to a document, where it can
fail, and what is deliberately not finished.

---

## Shape of the system

```
backend/
├── main.py               FastAPI app factory, middleware order, lifespan
├── api/
│   ├── dependencies.py   Service container — where everything is wired
│   ├── security.py       Bearer auth, CurrentUser, role checks
│   ├── middleware.py     Correlation ID, security headers, rate limit, metrics
│   └── routers/          HTTP surface: documents, chat, auth, system
├── services/             Business logic
├── repositories/         The only code that touches the ORM
├── core/                 Config, database engine, ORM models
├── models/               Pydantic request/response schemas
├── observability/        Structured logging, Prometheus metrics
└── tests/                Test suite
```

Dependencies point inward: routers depend on services, services depend on
repositories, repositories depend on the ORM. Nothing points back out.

### The service container

`api/dependencies.py` builds every service once at startup and stores the
container on `app.state`, not in a module global. Two application instances in
one process (which is what the test suite does) therefore do not share — or
tear down — each other's services.

The important wiring is this:

```python
retrieval_service  = RetrievalService()          # ONE instance
document_processor = DocumentProcessorService(retrieval_service=retrieval_service)
chat_service       = ChatService(retrieval_service=retrieval_service)
```

Both also take the *initialised* instance: they call
`retrieval_service.ensure_store_ready()`, never `initialize()`. Calling the
full one here would load the embedding model, undoing the deferral the
container chose on the line before — see "Startup sequence" below.

`DocumentProcessorService` and `ChatService` previously each constructed their
own `RetrievalService`, which meant each got its own `EmbeddingService` and its
own ChromaDB client. Documents indexed at upload went into one vector store;
chat searched a different, empty one. Every RAG answer returned zero sources
while every API call reported success. **If you change this wiring, that bug
comes back** — `test_document_workflow.py::TestRetrievalSeesUploadedDocuments`
is the guard.

---

### Startup sequence

The order matters, and it is the result of measurement rather than taste:

1. **Import.** `sentence_transformers` is imported lazily, not at module
   scope. Importing it pulls in torch and transformers and costs ~4s, and that
   was being paid during `import backend.main` — before uvicorn had even
   constructed the application, let alone run its lifespan. No amount of
   deferring the model *load* could reach it.
2. **Lifespan.** Database (migrations), then the vector store, then the
   services. All cheap and local: ~0.6s.
3. **Serving.** The lifespan returns, uvicorn accepts connections, and
   `/api/health/live` answers — about **1.5s** from process start, against
   7.4s before.
4. **Model load.** Started *last*, as a background task, because a task only
   runs when the loop gets control: one created earlier is executed during the
   lifespan's own awaits (the LLM availability checks), and since the import
   that follows is CPU-bound Python holding the GIL, the lifespan then waits
   for most of it anyway. Created before those checks, `initialize_services`
   took 5.48s; created after them, ~0.6s.

While the load is in flight, `/api/health/ready` returns 503 and a request
that needs embeddings **waits** rather than falling back to keyword matching.
That is not caution: a lexical vector and a model vector are points in
unrelated spaces, so a cosine similarity between them is meaningless. Indexing
with one and querying with the other would not degrade retrieval, it would
corrupt it — silently, since every score still comes back a plausible number.

`REQUIRE_REAL_EMBEDDINGS=true` loads inline instead. That setting exists so a
missing model is a startup failure; deferred, the failure would land on a task
nobody awaits and the process would come up reporting itself healthy.

---

## What happens to a document

```
POST /api/documents/upload
  │
  ├─ authenticate ───────────────► 401 if no valid token
  ├─ validate filename/extension ► 400 on an unsupported type
  ├─ stream to disk, capped ─────► 413 the moment the cap is passed
  ├─ sniff magic bytes ──────────► 400 if content contradicts the extension
  ├─ INSERT document (queued)
  └─ 202 Accepted { document_id }        ← returns here

background task
  │
  ├─ 10%  extract text ──────────► FAILED with a readable reason
  ├─ 40%  analyse ───────────────► entities, PII flags and an extractive
  │                                summary always; LLM prose when reachable
  ├─ 75%  chunk → embed → index ─► FAILED if indexing fails
  └─ 100% COMPLETED
```

Each step writes progress to the database and pushes it over the websocket, so
progress survives a page refresh and is visible to every worker.

### Retrieval and citation

Chunks are stored twice, on purpose:

- **Vector store (ChromaDB)** — the embedding, for similarity search.
- **`document_chunks` table** — the text, page number and character offsets.

The vector store is a derived index that can be rebuilt; the citation metadata
is the record of provenance. Keeping page numbers in the database means a
citation still resolves to "page 7 of contract.pdf" after a re-index.

### Ranking

Retrieval over-fetches candidates and re-orders them before truncating to
`top_k`. Vector similarity supplies one ordering; BM25 over the candidate pool
supplies the other, and the two are combined as a weighted sum after
normalising BM25 to `[0, 1]`.

The reason is that vector search ranks by meaning, which is what makes
paraphrased questions work and is exactly why it can rank a passage *about*
invoices above the one containing `INV-2025-1001`. Keyword scoring has the
opposite blind spot.

Term rarity is measured *within the retrieved pool*, so no corpus-wide index
has to be maintained across uploads and deletes. The consequence is that this
improves precision but not recall: a chunk the vector search never returned
cannot be recovered by re-ranking. A real BM25/FTS index over
`document_chunks`, unioned with the vector hits, is the upgrade path.

Finally, a **cross-encoder** reads the question and each surviving passage
*together* and scores the pair. A bi-encoder cannot do this: the passage vector
is computed before any question exists, so it cannot emphasise the part a
particular question is about. Cross-encoders are far too slow to run over a
corpus and exactly right for reordering a shortlist.

`SearchResult.score` keeps the vector similarity through all of this, so
`RETRIEVAL_MIN_SCORE` keeps its meaning and a citation still reports a
comparable number. Cross-encoder outputs are unbounded logits, not
similarities; writing them into `score` would quietly break both.

### Measured

`scripts/eval_retrieval.py` asks 13 paraphrased questions whose correct source
document is known, against the whole 22-document sample corpus. The questions
avoid the documents' own wording — "how much do we owe Quantum Industrial
Systems" has to reach "TOTAL DUE" — because matching a shared noun is not
evidence that retrieval works.

| Ranking | recall@1 | recall@3 | MRR |
|---|---|---|---|
| Semantic only | 62% | 77% | 0.699 |
| + keyword (hybrid) | 69% | 92% | 0.827 |
| + cross-encoder | 77% | 100% | 0.859 |
| Both | 77% | 100% | 0.859 |

Two things worth reading off that table. The cross-encoder runs last and
reorders everything, so **hybrid adds nothing on top of it** — the last two
rows are identical. Hybrid earns its place as the fallback: without
sentence-transformers there is no cross-encoder, and hybrid is then the
difference between 69% and 62%.

Set `RERANK_RESULTS=false` to skip the cross-encoder, or `HYBRID_RETRIEVAL=false`
to rank on similarity alone.

PDF extraction emits `--- Page N ---` markers, and
`retrieval_service._page_for_offset` maps a chunk's character offset back to
its page. That is the whole mechanism behind verifiable citations.

### Answering a question

```
POST /api/chat
  ├─ guardrails validate the input
  ├─ look up the caller's document IDs        ← the isolation boundary
  ├─ retrieve top-k, filtered by owner and score threshold
  ├─ assemble context, each chunk labelled [Source N: file, page P]
  ├─ ask the LLM
  └─ respond with grounded, embeddings_are_real and per-source citations
```

Retrieval is restricted to documents the caller owns. Without that filter a
user could ask questions whose answers are drawn from another tenant's files —
the retrieval layer would happily supply them.

`grounded` means *the answer text was built from the retrieved passages* — not
merely that retrieval returned something. If generation fails there is no
answer to ground, so `grounded` is false even though sources are present. The
LLM service signals failure by raising `LLMUnavailableError` rather than
returning an apology string; when both were plain strings the two were
indistinguishable, and a "model is unavailable" notice was shipped as a
`grounded: true` answer with a full citation list beside it.

If nothing clears the threshold, the response sets `grounded: false` **and**
prepends a sentence saying the answer is not supported by the documents. An
ungrounded answer that looks identical to a grounded one is how RAG products
mislead people.

---

### Comparing two documents

`GET /api/documents/compare` diffs the stored text of two documents the caller
owns. Paragraphs are the unit: a line diff of reflowed text reports every line
as changed when one word moved, and a word diff loses the context that makes a
change legible. Whitespace is normalised first for the same reason.

Each reported change quotes both sides and carries the page it is on, so it can
be checked against the originals the way a citation can. Nothing here goes
through the model, so this feature does not degrade when Ollama is absent and
cannot report a change that is not in the text.

---

## Data model

| Table | Holds | Notes |
|---|---|---|
| `documents` | Lifecycle, extracted text, AI analysis, checksum, storage path | Indexed on `(owner_id, status)` and `(owner_id, created_at)` |
| `document_chunks` | Chunk text, page, offsets | `ON DELETE CASCADE`; SQLite needs `PRAGMA foreign_keys=ON`, which the engine sets |
| `chat_feedback` | Thumbs up/down per message | Rows, not counters, so quality can be sliced by time and user |

Every document carries a SHA-256 of the uploaded bytes: it identifies
re-uploads and anchors provenance for anything derived from the file.

### Ownership

Enforced in `DocumentRepository`, not in the routers. Every read that can be
attributed to a user takes an `owner_id` and filters on it, so a new endpoint
cannot leak another tenant's documents by forgetting a check. A document
belonging to someone else returns **404, not 403** — a 403 would confirm that
the ID exists.

### Schema changes

Alembic, in `migrations/`. `DatabaseManager` applies pending revisions during
startup when `DB_AUTO_MIGRATE` is true, which is the default because a
single-container deployment has nowhere else to run them. Set it false and run
`alembic upgrade head` as a deploy step wherever more than one replica exists;
startup then verifies the tables are present and refuses to start if they are
not.

This replaced `Base.metadata.create_all`, which creates missing *tables* and
nothing else — so a new column reached a developer's fresh database, passed CI
(which starts from an empty SQLite file every run) and did nothing at all on
any database that already existed. It failed at query time instead, on the
deployment that mattered most.

Three details that are load-bearing:

- **Adoption.** A database built by the old `create_all` path has every table
  and no `alembic_version` row. Startup recognises that and stamps it at the
  baseline rather than replaying the baseline onto tables that exist. It
  requires **all** the baseline tables, not any of them: a database with only
  some is an interrupted bootstrap, and stamping it would skip the baseline so
  the missing tables were never created. That case is refused by name, because
  neither creating the absent tables nor dropping the present ones is safe to
  do unattended.
- **A failed migration is not a won race.** When a migration raises, startup
  continues only if the database has actually reached head — not merely if the
  tables are present. Those were the same condition under `create_all`, where
  a present table *was* success; with migrations they are not, and taking
  table presence as proof would start the application against a schema a
  failed revision never updated.
- **Batch mode.** `render_as_batch=True` in `migrations/env.py`, because SQLite
  cannot `ALTER` a column in place — it is the only way a column change reaches
  the default database. A no-op on PostgreSQL.
- **Concurrency.** Alembic's `EnvironmentContext` communicates through
  module-level globals, so two migrations in one interpreter tear down each
  other's state. A per-event-loop lock serialises them; separate Gunicorn
  worker *processes* are covered by the existing retry-and-recheck.

---

## Failure modes, and what each one does

The design principle: **degrade visibly, never silently**.

| What fails | Behaviour | How you find out |
|---|---|---|
| No embedding model | Keyword-only lexical matching | `/api/health` → `degraded`; `embeddings_are_real: false` on every response; `roneira_embedding_backend_real` gauge = 0 |
| tesseract missing | Scans and images are refused, naming the reason; text documents unaffected | The document's failure reason; `ocr_unavailable_reason` in metadata |
| Two workers migrate the schema at once | The loser waits and rechecks instead of dying | `Schema was migrated concurrently by another worker` at INFO |
| Embedding model still loading | Liveness 200, readiness 503; a search waits for the model rather than answering from incomparable vectors | `/api/health` → `starting`, `embeddings: starting` |
| `DB_AUTO_MIGRATE=false` and the schema is absent | Startup refuses, naming the missing tables | The startup error |
| A partially created schema (interrupted bootstrap) | Startup refuses, naming what is missing and what is present | The startup error |
| sentence-transformers installed but unimportable | Keyword-only matching, and the load settles so readiness still clears | `/api/health` → `degraded` with the import error; an ERROR-level log with the traceback |
| Ollama unreachable | Text still extracted and indexed; no summary or chat prose | `/api/health` → `degraded` with the endpoint it tried |
| Database down | Requests fail | `/api/health/ready` → 503; liveness stays 200 so the container is not killed |
| Text extraction fails | Document marked `failed` with a readable reason; nothing indexed | Document status; `roneira_documents_processed_total{status="error"}` |
| Retrieval matches nothing | Answer says so; `grounded: false` | `roneira_retrieval_queries_total{outcome="empty"}` |
| LLM unreachable *during* a chat | `grounded: false`, the retrieved passages are still returned as sources so they can be read directly | `/api/health` → `degraded`; the answer text says the model is not running |
| Indexing fails | Document marked `failed`, not `completed` | Document status |

The lexical fallback deserves emphasis. When `sentence-transformers` is not
installed, `EmbeddingService` builds vectors with the hashing trick — hashed
bag-of-words with sub-linear term frequency, L2-normalised. That is genuine
keyword matching, so the system remains usable and CI stays meaningful without
a multi-gigabyte model download. It is **not** semantic: paraphrases and
synonyms will not match. Set `REQUIRE_REAL_EMBEDDINGS=true` to make it a
startup failure instead.

An earlier implementation tiled 32 hash bytes across all 384 dimensions, so
unrelated texts correlated arbitrarily and search returned noise while
reporting success.

---

## Middleware order

Starlette runs middleware outermost-first. The order in `create_application`
is deliberate:

1. **CorrelationIdMiddleware** — first, so everything downstream logs with an
   ID. Honours an inbound `X-Correlation-ID` (truncated to 128 characters so a
   client cannot write unbounded data into the logs).
2. **SecurityHeadersMiddleware** — outside the routers, so headers apply to
   error responses too.
3. **MetricsMiddleware** — labels by *route template*, not concrete path;
   document IDs as label values would give Prometheus unbounded cardinality.
4. **TrustedHost** — only installed when `ALLOWED_HOSTS` is actually
   restrictive.
5. **CORS**.
6. **RateLimitMiddleware** — tighter budget for upload, chat and search, since
   each costs a document parse or an LLM inference.

Rate-limit counters are per process. With N replicas the effective limit is
N × the configured value. For a hard global limit, enforce it at the ingress.

---

## Configuration

`backend/core/config.py` is the single source of truth. If a variable is not a
field there, it has no effect — no matter what any document says.

Note for anyone editing it: pydantic-settings v2 **ignores** `Field(env="...")`.
Environment names that differ from the field name need an explicit
`validation_alias`. `JWT_ALGORITHM` and `APP_VERSION` silently did nothing for
exactly this reason.

Setting `ENVIRONMENT=production` runs a validator that refuses to start with a
placeholder secret key, a key under 32 characters, `REQUIRE_AUTHENTICATION=false`,
wildcard CORS or hosts, or `DEBUG=true`.

---

## Unwired modules

`backend/training/` exports offline training-data/configuration utilities;
it is not a training pipeline or an application request path. The optional
`backend/observability/tracing.py` shim is also outside the normal coverage
scope. Keep these explicit exports until their offline consumers are known.

Entity extraction, PII flags, and extractive summaries are active through
`services/document_insights.py`. Its PII adapter returns types, counts, and
offsets rather than persisting matched personal values. Feedback uses the
persisted `FeedbackRepository`, not in-memory service counters.

`research/` contains the separate data-catalogue, synthetic-data, ontology,
orchestration, and evaluation utilities; `pytest tests/ --no-cov` covers them.
They are not imported by the web application. Retained demo scripts are
examples/illustrations, not measured production or performance evidence.
`python scripts/demo_task1_catalog.py` recreates the ignored Task 1 HTML
outputs under `docs/roadmap/screenshots/`.

## Dependencies

Keep direct requirements tied to actual runtime, script, test, or tooling
usage. Install from `requirements.txt` and `package-lock.json`; do not carry
speculative integrations or upgrade unrelated packages as cleanup.

- Alembic manages schema revisions; asyncpg supplies the PostgreSQL driver.
- Chroma is embedded through `PersistentClient`, not exposed as a Chroma HTTP
  server. Known server-advisory tracking remains in
  [#46](https://github.com/aaron-seq/Roneira-AI-LLM-powered-document-intelligence-system/issues/46).
- sentence-transformers supplies embeddings and reranking; model failures
  must remain visible through health and response fields.
- Ruff owns Python formatting/lint rules. Mypy follows the executing Python
  version so version-specific dependency stubs match the CI interpreter.
- Bundle analysis is optional; see
  [CONTRIBUTING.md](../CONTRIBUTING.md#before-you-open-a-pull-request).

---

## Known limitations

Stated plainly so nobody discovers them the hard way:

- **OCR needs a system binary.** Scanned PDFs and image uploads are read with
  tesseract, which pip cannot install. Without it OCR reports itself
  unavailable and scans are refused with that reason — they are never indexed
  as empty. Accuracy is whatever tesseract gives on the page as scanned; there
  is no deskewing or denoising pass.
- **Tables are partially handled.** `.docx` table cells are extracted; PDF
  tables are flattened into text. `extract_tables` is accepted as an upload
  option and currently ignored.
- **Single-node vector store.** ChromaDB persists to local disk. More than one
  worker means more than one process writing the same directory; run one
  worker, or move to a shared vector service.
- **Conversation memory is in-process.** Chat history is lost on restart and
  is not shared between workers. Documents are not — those are in the database.
- **Rate limiting is per process** (see above).
- **The user store is a dictionary.** Two built-in accounts, no registration,
  no password reset. See [SECURITY.md](SECURITY.md).
- **Streaming is simulated.** `stream_chat` chunks a completed answer rather
  than streaming from the model.
- **No background job queue.** Processing runs in a FastAPI background task, so
  a restart mid-document leaves it stuck in `processing`.

---

## Testing

The [backend suite](../backend/tests/) uses a real application instance with
disposable SQLite, vector storage, and uploads. Coverage exclusions are
explicit in [.coveragerc](../.coveragerc), with a 70% gate. The
[test strategy](testing/TESTING_GUIDELINES.md) describes fixtures, failure and
ownership assertions, optional-dependency skips, and remaining verification
gaps. Commands and the platform matrix live in
[CONTRIBUTING.md](../CONTRIBUTING.md#before-you-open-a-pull-request).
