# Testing Guidelines & Best Practices

Commands and the CI matrix live in
[CONTRIBUTING.md](../../CONTRIBUTING.md#before-you-open-a-pull-request) and the
[workflow](../../.github/workflows/ci.yml).

## Scope and fixtures

- `backend/tests/` exercises the application with a disposable SQLite database,
  vector store, and upload directory. The session fixture settles model loading;
  startup tests separately exercise loading and failure windows.
- `tests/` covers the independent `research/` utilities. Run it explicitly;
  its coverage is separate from the backend's 70% gate.
- CI starts a real HTTP server for authenticated upload, processing, search,
  and anonymous-denial checks; the Docker job probes liveness and readiness.
- `npm test` uses Node's built-in test runner for dashboard feedback and health
  indicators; no frontend testing dependency is required.
- A missing optional model/OCR binary may skip dependency-specific cases.
  Report those skips and distinguish cached/offline, simulated, and real
  integrations. There is no configured browser E2E suite or performance SLA.

## Test design

Start from a requirement, invariant, or reproduced defect. Assert observable
results and durable state, including forbidden side effects. For ownership
denials, verify the intended actor/resource and that no data leaked or changed.

Cover relevant boundaries: empty/malformed input, limits, expired/invalid
credentials, dependency failures, retries, concurrency, and cancellation.
Mock external boundaries without mocking the control being tested. Use
synthetic data and deterministic fixtures; keep credentials and user data out
of logs and artifacts.

For behavior fixes, show the regression fails for the intended cause before
the fix and passes afterward. For file/prose cleanup, use import, build,
example, link, and packaging checks instead of artificial unit tests. Preserve
meaningful assertions and thresholds; do not suppress failures for a green run.

Record the revision, environment, exact commands, result, skips, and evidence
in the issue/PR. Coverage and scanner counts identify gaps; they do not prove
security, accessibility, performance, or production readiness.
