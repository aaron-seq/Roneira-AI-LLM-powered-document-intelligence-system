# Roneira — Document Intelligence

Self-hosted document extraction, search, comparison, and question answering
with file/page citations. The backend is FastAPI; the frontend is React.

[![CI](https://github.com/aaron-seq/Roneira-AI-LLM-powered-document-intelligence-system/actions/workflows/ci.yml/badge.svg)](https://github.com/aaron-seq/Roneira-AI-LLM-powered-document-intelligence-system/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What it does

- Extracts text from PDFs, Word documents, text files, and scanned pages.
- Indexes passages for semantic/keyword search and returns source citations.
- Produces extractive summaries, entity/PII flags, and optional Ollama answers.
- Compares two documents and exports their text and analysis as Markdown.
- Restricts document access and retrieval to the authenticated owner.

### What it is not

The bundled accounts are for development. There is no persistent user store,
SSO, audit log, or complete protection against prompt injection. Read
[security and data handling](docs/SECURITY.md) before using real documents.
Multiple workers/replicas also need storage changes described in
[deployment](docs/DEPLOYMENT.md#scaling).

## Quick start

Use Python 3.11–3.13 (the CI matrix) and Node 22.12+ for the frontend.
Ollama and the system OCR binary are optional.

```bash
git clone https://github.com/aaron-seq/Roneira-AI-LLM-powered-document-intelligence-system.git
cd Roneira-AI-LLM-powered-document-intelligence-system
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn backend.main:app --reload
```

On PowerShell, activate with `.venv\Scripts\Activate.ps1`.
Open <http://localhost:8000/api/docs>. SQLite, migrations, and the local vector
store initialize on first run; no database server is needed for development.

<a id="try-it-in-60-seconds"></a>
### First upload and search

This Bash example uses the built-in development account:

```bash
TOKEN=$(curl -fsS -X POST http://localhost:8000/api/auth/token \
  -d "username=demo&password=demo" | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
DOC=$(curl -fsS -X POST http://localhost:8000/api/documents/upload \
  -H "Authorization: Bearer $TOKEN" -F "file=@docs/samples/INV-2025-1001.pdf" \
  | python -c "import sys,json; print(json.load(sys.stdin)['document_id'])")
curl -fsS "http://localhost:8000/api/documents/$DOC/status" -H "Authorization: Bearer $TOKEN"
```

Processing is asynchronous. Once the status is `completed`, search:

```bash
curl -fsS -X POST http://localhost:8000/api/search \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"query":"total amount due","top_k":3}'
```

### Or load the whole sample corpus at once

```bash
python scripts/load_samples.py
```

The loader uploads and waits for the 22 synthetic documents in
[docs/samples](docs/samples/README.md), which also lists useful questions.

### Turning on semantic search

`requirements.txt` includes sentence-transformers. Its embedding and
reranking models may download on first use. `/api/health/ready` returns 503
while loading; requests needing embeddings wait to avoid mixing incompatible
vectors. If loading fails, responses declare keyword fallback.
`REQUIRE_REAL_EMBEDDINGS=true` instead makes model availability a startup
requirement. See [ranking and its evaluation](docs/ARCHITECTURE.md#ranking).

### Reading scanned documents

Install Tesseract separately (`brew install tesseract`,
`sudo apt-get install tesseract-ocr`, or the
[Windows installer](https://github.com/UB-Mannheim/tesseract/wiki)). Verify it:

```bash
python -c "from backend.common.ocr import ocr_availability; print(ocr_availability())"
```

Use `TESSERACT_PATH` if it is outside `PATH`, or `ENABLE_OCR=false` to disable
OCR. Pages with empty or near-empty text layers are OCR'd. Inspect `ocr_pages` and the
original scan before relying on extracted text; handwriting/deskew/denoising
are unsupported.

### Turning on summaries and chat

Follow [Ollama setup](docs/OLLAMA_SETUP.md). Without Ollama, extraction,
extractive summaries, indexing, search, and comparison still work; generated
prose is unavailable and health/response fields report the limitation.

### With Docker

```bash
cp .env.example .env
docker compose up
docker compose --profile observability up   # optional Prometheus/Grafana
```

The development API is on `:8000`, frontend on `:3000`, Prometheus on `:9090`,
and Grafana on `:3001`. See [deployment](docs/DEPLOYMENT.md) for production
configuration, probes, logs, backups, and migrations.

### Frontend

```bash
npm ci
npm run dev
```

Open <http://localhost:3000> with the backend running. Vite proxies `/api` to
the backend; `VITE_API_URL` controls the client URL at build time.

## API

The running application's [OpenAPI reference](http://localhost:8000/api/docs)
is the endpoint/schema reference. Upload is `POST /api/documents/upload`
(202); status is `GET /api/documents/{document_id}/status`; search and chat
are `POST /api/search` and `POST /api/chat`. Authenticate through
`POST /api/auth/token` and send `Authorization: Bearer <token>`.
Another owner's document returns 404 without confirming its existence.

### Reading a document's summary

`ai_insights.summary_source` is `llm`, `extractive`, or `verbatim`. The last
means the document was too short to compress, so exports label it document
text rather than a summary.

### Reading a chat response

Check `grounded`, `embeddings_are_real`, and each source's `filename`,
`page_number`, `score`, and `content_preview`. `grounded: false` means there
is no supported generated answer; sources may still be returned for inspection.

## Configuration

[.env.example](.env.example) is the annotated configuration reference;
[backend/core/config.py](backend/core/config.py) defines accepted settings.
Production startup rejects placeholder/short secrets, disabled authentication,
wildcard origins/hosts, and debug mode. Retention, OCR, retrieval, model, and
migration settings are documented beside their values.

## Architecture

Routers delegate to services and repositories. Processing and chat share one
retrieval service; documents/chunks persist in SQL and the derived index in
Chroma. See [architecture](docs/ARCHITECTURE.md) for data flow, ownership,
startup, failure modes, offline modules, and limitations.

## Development

```bash
pytest
pytest tests/ --no-cov
npm run type-check && npm run lint && npm run build
```

[CONTRIBUTING.md](CONTRIBUTING.md) lists the full lint/type/migration checks,
test conventions, and optional bundle analysis. CI also exercises authenticated
upload/search and the production backend image.

## Troubleshooting

Start with `/api/health` and its component reasons. For missing models, OCR,
dependency mismatches, Windows issues, and startup failures, use the
[troubleshooting table](CONTRIBUTING.md#troubleshooting) and
[Ollama guide](docs/OLLAMA_SETUP.md#troubleshooting).

## Roadmap

Tracked work is in [GitHub issues](https://github.com/aaron-seq/Roneira-AI-LLM-powered-document-intelligence-system/issues).
The following directions are proposals, not implemented capabilities:

### Next

Table/structured-field extraction and real streaming chat.

### After that

Collections/tags, redaction before indexing, and a persistent user store.

### Longer term

Shared workspaces, labelled retrieval evaluation, durable background jobs, and
document-source connectors. See [known limitations](docs/ARCHITECTURE.md#known-limitations).

## Documentation

| Guide | Purpose |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | Runtime contracts, data, failure modes |
| [Security](docs/SECURITY.md) | Controls, data handling, reporting gap |
| [Deployment](docs/DEPLOYMENT.md) | Scaling, recovery, observability, migrations |
| [Ollama](docs/OLLAMA_SETUP.md) | Optional local inference |
| [Samples](docs/samples/README.md) | Synthetic corpus and questions |
| [Contributing](CONTRIBUTING.md) | Checks and conventions |
| [Project status](docs/product/PROJECT-STATUS.md) | Current GitHub work and evidence |

## License

MIT — see [LICENSE](LICENSE).
