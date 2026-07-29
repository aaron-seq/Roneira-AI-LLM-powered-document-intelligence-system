# Security

What this system defends against today, what it does not, and what you must
change before putting real documents in it.

---

## Read this first

**The bundled `demo` and `admin` accounts are a development convenience, not a
user management system.** They live in a Python dictionary in
`backend/services/auth_service.py`, with passwords that default to `demo` and
`admin123`. Anyone who can reach the API and has read the README can sign in.

Before this handles documents you would not publish:

1. Set `SECRET_KEY` to a real random value (≥32 characters). Anyone with it can
   mint tokens for any user.
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(48))"
   ```
2. Set `ENVIRONMENT=production`. The service will then **refuse to start** with
   a placeholder key, disabled authentication, or wildcard CORS.
3. Replace `AuthService._build_default_users` with a real user store, or put
   the API behind an identity-aware proxy that authenticates before traffic
   arrives.
4. Set `ALLOWED_ORIGINS` and `ALLOWED_HOSTS` to your actual domains.
5. Terminate TLS in front of the service. Bearer tokens over plain HTTP are
   readable by anything on the path.

Until step 3, treat this as single-tenant software for people you already
trust.

---

## What is enforced

### Authentication and authorization

- Every document and chat endpoint requires a valid bearer token. Public
  endpoints are exactly: `/api/`, `/api/health`, `/api/health/live`,
  `/api/health/ready`, `/api/metrics`, `/api/documents/formats/supported`, and
  the token endpoint itself.
- JWTs are signed with a pinned algorithm. Pinning is what stops a forged
  token that claims `alg: none`.
- Tokens are typed (`type: access`) and checked, so a future refresh token
  cannot be replayed as an access token.
- Ownership is filtered in the repository layer, not in individual routes.
- A document owned by someone else returns **404, not 403**: a 403 confirms the
  ID exists, which is itself a leak.
- Login returns one message for every failure and equalises its timing with a
  dummy hash comparison, so the endpoint is not a username oracle.
- Passwords are hashed with PBKDF2-SHA256.

Regression tests: `backend/tests/test_auth.py`, and the
`TestOwnershipIsolation` class in `test_document_workflow.py`.

### File uploads

Validated in this order, which is the order that protects the server:

1. Extension against an allow-list — cheapest rejection first.
2. Bytes streamed to disk under a hard cap, aborting **mid-transfer** with 413.
   The previous implementation read the entire file into memory and then
   checked its size, so an unauthenticated caller could force an arbitrary
   allocation.
3. Magic-byte sniffing (libmagic where available, a signature table otherwise)
   checked against the extension. A renamed executable is rejected.
4. Empty files rejected.

Storage names are derived from a generated UUID, never from the uploaded
filename, and path separators in filenames are rejected outright.

Source files are served through an authorized route with
`Content-Disposition: attachment` — never a public static mount. Serving
user-uploaded HTML or SVG inline from the same origin would be stored XSS.

Regression tests: `backend/tests/test_upload_validation.py`.

### Transport and headers

Applied to every response, including errors:

`X-Content-Type-Options: nosniff` · `X-Frame-Options: DENY` ·
`Referrer-Policy: strict-origin-when-cross-origin` ·
`Cross-Origin-Opener-Policy: same-origin` · `Permissions-Policy` (camera,
microphone, geolocation denied) · a restrictive `Content-Security-Policy`
(`default-src 'none'; frame-ancestors 'none'`, since the API serves JSON).

HSTS is added only when `ENVIRONMENT=production`, because pinning HSTS on a
plain-HTTP local run leaves developers with a browser that refuses to connect.

### Rate limiting

Per-client fixed-window budgets: `RATE_LIMIT_PER_MINUTE` for reads,
`RATE_LIMIT_EXPENSIVE_PER_MINUTE` for upload, chat and search. Health and
metrics are exempt so infrastructure polling is never throttled.

Clients are keyed by authenticated subject where present, otherwise peer
address. `X-Forwarded-For` is **deliberately not trusted**: unless your proxy
is known to overwrite it, any client can set it and reset its own budget. If
you run behind a trusted proxy, configure the ASGI server's proxy headers
instead.

### Error handling

Unhandled exceptions return a generic message plus a correlation ID; the stack
trace goes to the logs only. Exception text has leaked file paths, SQL and
connection strings in more products than anyone would like.

Correlation IDs are echoed in `X-Correlation-ID`, so a user-reported error can
be found in the logs without guesswork.

---

## What is not defended against

Be clear-eyed about these:

| Gap | Consequence | Mitigation today |
|---|---|---|
| **Prompt injection** | Text inside an uploaded document can attempt to steer the model. Retrieved chunks are fenced and labelled `[Source N: …]`, which helps but is not a guarantee. | Treat generated output as untrusted. Do not wire it to actions. |
| **No document encryption at rest** | Retained originals and extracted text sit in plain files and database rows. | Use encrypted volumes/disks. Set `RETAIN_SOURCE_FILES=false` where originals must not persist. |
| **No audit log** | There is no durable record of who read which document. | Ship the access logs (they carry correlation IDs and paths) to a retained store. |
| **No PII detection on ingest** | `pii_detection_service.py` exists but is not wired in; documents are indexed as-is. | Do not upload documents whose PII must never be indexed. On the roadmap. |
| **Archive/zip contents unscanned** | `.docx` is a zip container; malformed archives are handed to `python-docx`. | Keep dependencies patched; run `pip-audit`. |
| **No malware scanning** | Uploaded files are stored and can be downloaded back. | Scan the upload directory out of band if untrusted users can upload. |
| **Token revocation** | JWTs are valid until they expire; there is no denylist. | Keep `ACCESS_TOKEN_EXPIRE_MINUTES` short. |
| **Rate limits are per process** | N replicas means N × the limit. | Enforce a global limit at the ingress. |

---

## Data handling

- **What is stored:** the original file (unless `RETAIN_SOURCE_FILES=false`),
  the extracted text, chunk text with page offsets, embedding vectors, AI
  analysis, and a SHA-256 of the upload.
- **Where:** the configured database, `UPLOAD_DIRECTORY`, and
  `VECTOR_STORE_PATH` — all local by default.
- **What leaves the machine:** nothing. Inference runs against Ollama on a host
  you control. There is no telemetry and no third-party API call.
- **Deletion:** `DELETE /api/documents/{id}` removes the database row, the
  chunk rows, the vectors, and the stored file. It is not a soft delete.
- **Retention:** `SOURCE_RETENTION_DAYS` bounds how long originals live;
  `DocumentProcessorService.purge_expired_sources()` performs the sweep.
  Note that this is not yet on a scheduler — call it from a cron job.

---

## Running the security checks

```bash
pip install bandit pip-audit
bandit -r backend/ -ll -x backend/tests
pip-audit --requirement requirements.txt
```

Both run in CI on every pull request.

---

## Reporting a vulnerability

Please **do not** open a public issue for a security problem.

Use GitHub's private vulnerability reporting on this repository
(Security → Report a vulnerability), or contact the maintainer directly.
Include reproduction steps and affected versions. You can expect an
acknowledgement within a week.

This is a personal open-source project, not a commercial product with an
on-call rotation — please calibrate expectations accordingly, and do not use
it for anything where a delayed patch would be unacceptable.
