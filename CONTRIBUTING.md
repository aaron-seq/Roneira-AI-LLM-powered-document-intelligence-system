# Contributing

Thanks for considering it. This guide is short on purpose — the goal is to
make it easy to land a good change, not to enumerate rules.

---

## Setup

```bash
git clone https://github.com/aaron-seq/Roneira-AI-LLM-powered-document-intelligence-system.git
cd Roneira-AI-LLM-powered-document-intelligence-system

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install ruff mypy                 # dev tooling

cp .env.example .env
pytest                                # should pass before you change anything
uvicorn backend.main:app --reload
```

Frontend:

```bash
npm install
npm run dev
```

If `pytest` does not pass on a clean checkout, that is a bug — please open an
issue rather than working around it.

---

## Before you open a pull request

```bash
pytest                                                  # tests + 70% coverage gate
ruff check backend/ && ruff format backend/
mypy backend/core backend/api backend/repositories --ignore-missing-imports

npm run type-check && npm run lint && npm run build
```

CI runs all of this plus a smoke job that boots the app and drives a document
from upload through to a successful search.

---

## What a good change looks like

**Every behaviour change comes with a test.** Not for a coverage number — for
the next person, who needs to know whether they just broke something. The most
valuable tests in this repository are the ones that encode a bug we already
shipped:

- `TestRetrievalSeesUploadedDocuments` — uploads were invisible to search
- `TestOwnershipIsolation` — every user could read every document
- `TestDocumentsSurviveRestart` — all state lived in a dict
- `test_failure_messages_never_masquerade_as_content` — extraction errors were
  indexed as document text

Write the test that would have caught your bug, then fix it.

**Say why in comments, not what.** The code says what it does. A comment
earns its place by explaining a constraint or a decision that is not obvious:

```python
# X-Forwarded-For is deliberately NOT trusted: unless the proxy is known to
# overwrite it, any client can set it and reset its own budget.
```

**Fail loudly, degrade visibly.** This system runs against optional
dependencies (an embedding model, an LLM) that may be absent. When a
capability is missing, the rule is: keep serving what still works, and make
the reduced capability visible in `/api/health`, in the API response, and in a
metric. Never return a confident-looking answer produced by a fallback the
caller cannot detect. That principle is why `embeddings_are_real` and
`grounded` appear on responses.

**Update the docs when behaviour changes.** Especially `.env.example` (which is
also the configuration reference) and `docs/ARCHITECTURE.md#known-limitations`.

---

## Project conventions

**Layering.** Dependencies point inward and nothing points back out:

```
routers  →  services  →  repositories  →  ORM
```

- Routers do HTTP: parse, authorize, delegate, shape the response.
- Services hold business logic and know nothing about HTTP.
- Repositories are the only code that touches the ORM. **Ownership filtering
  lives here**, so a new endpoint cannot leak another tenant's data by
  forgetting a check.

**Wiring** happens in `backend/api/dependencies.py`. Services are constructed
once at startup and injected. Do not construct a service inside another
service — that is exactly how the upload pipeline and chat ended up with
separate vector stores and no working retrieval.

**Configuration** goes in `backend/core/config.py` and nowhere else. Add the
field, document it in `.env.example`, and note that pydantic-settings v2
ignores `Field(env=...)` — use `validation_alias` when the environment name
differs from the field name.

**Style** is whatever `ruff format` produces. Line length 90. Type hints on
public functions. Google-style docstrings on anything non-obvious.

---

## Working on the unwired modules

`backend/` contains modules that are present but not imported by the running
application — OCR, PII detection, entity extraction, the multi-provider LLM
abstraction, and others. They are listed in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#unwired-modules).

Wiring one up is among the highest-value contributions available, and OCR is
the biggest gap between what the project promises and what it does. If you
take one on:

1. Remove its entry from `.coveragerc` and from the per-file ignores in
   `pyproject.toml`.
2. Bring tests with it — it will now be inside the coverage gate.
3. Add the failure mode to the table in `docs/ARCHITECTURE.md`, including how
   an operator finds out when it degrades.

---

## Reporting things

**Bugs** — include what you did, what you expected, what happened, and the
`X-Correlation-ID` from the response header if there was one. It appears in
the server logs and turns a vague report into a specific one.

**Security issues** — do not open a public issue. See
[docs/SECURITY.md](docs/SECURITY.md#reporting-a-vulnerability).

**Features** — describe the workflow you are trying to complete rather than
the implementation you have in mind. The roadmap in the README is a good place
to check whether it is already planned.

---

## Reviews

Expect questions about behaviour under failure: what happens when the model is
missing, the disk is full, the document is a scan, or two users hit the same
endpoint. Those questions are not obstruction — most of this project's history
of bugs lives in exactly those paths.

Small, focused pull requests get reviewed faster than large ones. If a change
needs a refactor first, that refactor is welcome as its own pull request.
