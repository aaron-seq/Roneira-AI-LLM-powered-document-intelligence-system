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

`DocumentProcessorService` and `ChatService` previously each constructed their
own `RetrievalService`, which meant each got its own `EmbeddingService` and its
own ChromaDB client. Documents indexed at upload went into one vector store;
chat searched a different, empty one. Every RAG answer returned zero sources
while every API call reported success. **If you change this wiring, that bug
comes back** — `test_document_workflow.py::TestRetrievalSeesUploadedDocuments`
is the guard.

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
  ├─ 40%  LLM enrichment ────────► degrades: text is kept, summary skipped
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

---

## Failure modes, and what each one does

The design principle: **degrade visibly, never silently**.

| What fails | Behaviour | How you find out |
|---|---|---|
| No embedding model | Keyword-only lexical matching | `/api/health` → `degraded`; `embeddings_are_real: false` on every response; `roneira_embedding_backend_real` gauge = 0 |
| tesseract missing | Scans and images are refused, naming the reason; text documents unaffected | The document's failure reason; `ocr_unavailable_reason` in metadata |
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

These exist in `backend/` and are **not imported by the running application**.
They are kept because several are worth adopting, but nothing calls them
today. Verified by walking the import graph from `backend.main`, not assumed:

| Module | What it would provide |
|---|---|
| `services/pii_detection_service.py` | PII detection and redaction on ingest |
| `services/entity_extraction_service.py` | Structured entity extraction |
| `services/summarization_service.py` | Multi-strategy summarization |
| `services/cross_reference_service.py` | Links between documents |
| `services/advanced_rag.py` | Hybrid search, reranking, query expansion |
| `services/request_batching.py` | Batched LLM inference |
| `services/sse_stream.py` | Server-sent events for streaming |
| `services/llm_providers/*` | Multi-provider abstraction (Azure OpenAI, Ollama) |
| `services/azure_service.py`, `free_llm_service.py` | Alternative backends |
| `training/*` | Fine-tuning data preparation |
| `common/utils.py` | Assorted helpers |

They are excluded from the coverage gate in `.coveragerc` and from the
stricter lint rules in `pyproject.toml`, both with comments pointing here.
**Wiring one up means removing its entry from both files and bringing tests
with it.**

There is also a legacy `app/` tree (an older Azure-oriented implementation)
and a `src/` tree of research and demo scripts. Neither is imported by
`backend.main`. `app/` is what the Dockerfile's production stage used to run,
which is how production and development ended up as different applications.

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

`pytest` runs the suite against a real application instance with a throwaway
SQLite database and vector store — the same wiring production uses, not a
parallel mock of it.

| File | Guards |
|---|---|
| `test_document_workflow.py` | Upload → process → index → retrieve → delete; tenant isolation; pagination |
| `test_persistence.py` | Documents survive a restart; source retention |
| `test_auth.py` | Every protected endpoint rejects anonymous and forged tokens |
| `test_upload_validation.py` | Renamed binaries, oversize, empty, traversal |
| `test_text_extraction.py` | Failures fail, and never become indexed content |
| `test_rag_grounding.py` | Page citations, context assembly, lexical fallback quality |
| `test_api_contract.py` | Health honesty, correlation IDs, headers, metrics, OpenAPI |
| `test_config.py` | Documented env vars work; production hardening refuses unsafe config |
| `test_progress_reporting.py` | Websocket progress shape; structured-logger call signatures |

Coverage is measured over code the application actually runs (see
`.coveragerc`), with a 70% gate. Including the unwired modules would drag the
number to roughly 47% and hide regressions in the code that does run.
