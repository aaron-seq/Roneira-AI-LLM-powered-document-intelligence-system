# Contributing

## Setup

Use the [README quick start](README.md#quick-start), including the pinned
backend requirements and `npm ci`. Keep local credentials, databases, uploads,
virtual environments, and build output outside version control.

## Before you open a pull request

```bash
pytest
pytest tests/ --no-cov
ruff check backend/ research/ migrations/
ruff format --check backend/ research/ migrations/
mypy backend/core backend/api backend/repositories --ignore-missing-imports --no-strict-optional --follow-imports=silent
alembic upgrade head
alembic check
npm run type-check && npm run lint && npm run build
```

Run migration commands against a disposable `DATABASE_URL`, never a shared
database. CI uses Python 3.11/3.12/3.13 for lint/types, exercises backend and
research tests, checks migration drift, drives an authenticated upload/search,
and builds/probes the backend image. The backend coverage gate is 70%.

For an optional local bundle report at `dist/stats.html`:

```bash
ANALYZE_BUNDLE=true npm run build
```

PowerShell: `$env:ANALYZE_BUNDLE='true'; npm run build`; clear the variable
before a normal release build. Normal builds omit this diagnostic report.

## What a good change looks like

- Test observable behavior and failure paths. Reproduce a defect before
  fixing it; preserve denied access, durable state, and side-effect assertions.
- Use synthetic data and isolated fixtures. Do not run tests against live
  user data. [Testing guidance](docs/testing/TESTING_GUIDELINES.md) describes
  the fixture boundaries and coverage gaps.
- Declare degraded capabilities in health and response fields; do not turn
  model failures into confident-looking content.
- Explain why a constraint exists. Avoid comments that repeat the code.
- Update affected configuration and documentation. Keep permanent guidance
  separate from PR-specific evidence and completion reports.

## Project conventions

Dependencies point inward: `routers → services → repositories → ORM`.
Routers parse/authenticate/delegate; repositories enforce ownership filtering.
Services are wired in `backend/api/dependencies.py`. Processing and chat must
share the same retrieval service so uploaded documents are searchable.

Settings belong in `backend/core/config.py` and [.env.example](.env.example).
With pydantic-settings v2, use `validation_alias` when the environment name
differs from the field name; `Field(env=...)` does not configure that mapping.

Schema changes need reviewed Alembic migrations:

```bash
alembic revision --autogenerate -m "describe the schema change"
```

Autogenerate cannot safely infer renames or data backfills. Read
[migrations/README.md](migrations/README.md) before applying generated code.

`src/` is the React application; `backend/` is the API; `research/` contains
separately tested Python utilities. Ruff defines Python style (line length 90).
Use existing ESLint/TypeScript settings for frontend changes.

## Working on the unwired modules

Offline training utilities and optional tracing remain outside the running
application's coverage scope. If wiring them into a request path, add behavior
tests and remove the relevant coverage omission in the same change. Entity
extraction, PII flags, summaries, and OCR are already active; see
[architecture](docs/ARCHITECTURE.md#unwired-modules).

## Troubleshooting

| Symptom | Check/action |
|---|---|
| Imports fail or Ruff rejects a configured rule | Activate the environment and reinstall `requirements.txt`; `pytest.ini` supplies the repository import path. |
| Python install fails with Windows `WinError 206` | Use a shorter virtual-environment path; PyTorch contains deeply nested paths. |
| Chat cannot generate prose | Check `/api/health`, `OLLAMA_BASE_URL`, `ollama serve`, and `ollama list`. |
| `embeddings_are_real` is false | Inspect the embedding component's health reason; weights/imports may be unavailable even when the package is installed. |
| A scan is rejected or garbled | Verify Tesseract and `TESSERACT_PATH`; inspect `ocr_pages`. There is no handwriting/deskew/denoise pass. |
| Production startup rejects secrets, CORS, hosts, or debug | Fix the named setting; keep production validation enabled. |
| Windows test cleanup reports a locked Chroma SQLite file | Chroma may retain a handle; the disposable fixture tolerates that cleanup limitation. |
| An older environment raises `np.float_` errors | Reinstall the current requirements, which pair Chroma 1.x with NumPy 2. |
| SQLite/Gunicorn startup fails with multiple workers | Keep `WORKERS=1`; PostgreSQL and shared storage changes are required before scaling. |
| Python crashes under Git Bash (`TP_NUM_C_BUFS`) | Use PowerShell or cmd for Python/pytest/uvicorn. |
| `localhost:3000` shows another app | Check port ownership or use `http://127.0.0.1:3000`. |

## Reporting things

For bugs, include steps, expected/actual behavior, revision, safe logs, and the
`X-Correlation-ID` if available. For security concerns, follow
[SECURITY.md](docs/SECURITY.md#reporting-a-vulnerability); do not publish
vulnerability details in public issues.

## Reviews

Keep changes cohesive. Report exact commands/revisions, distinguish local
checks from published CI, and disclose skipped or blocked checks. Review
questions should focus on behavior, failure handling, security boundaries,
compatibility, and maintenance cost.
