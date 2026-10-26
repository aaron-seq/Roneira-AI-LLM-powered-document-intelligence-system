# Engineering and agent reference

Use this repository's active user/project instructions, then the maintained
guides below. This reference does not grant deployment, disclosure, data
deletion, or repository-administration authority.

## Guardrails

- Establish the users, scope, API contract, and trust boundaries before
  substantial changes. Clarify missing decisions that affect security,
  authorization, data exposure, or irreversible actions.
- Do not expose databases publicly, hardcode credentials, disable TLS, or add
  protected operations without authentication and authorization.
- Avoid unbounded queries, blocking I/O on request hot paths, and N+1 access.
  Measure performance against actual requirements rather than invented SLAs.

## Design and implementation

- Reuse maintained libraries and existing project paths before adding
  abstractions or services. Build domain behavior; justify new dependencies
  and external services with their privacy, cost, and maintenance effects.
- Follow the [runtime architecture](ARCHITECTURE.md) and
  [contribution conventions](../CONTRIBUTING.md#project-conventions).
- Declare dependencies, keep environment configuration separate from code,
  and separate build/release/run. Document actual stateful-process limits.
- Use [C4/ADR guidance](architecture/BOOTCAMP_STANDARDS.md) when a decision
  needs a durable explanation, not for every routine edit.

## Verification and delivery

- Use isolated synthetic data, meaningful before/after assertions, negative
  access tests, and regressions that detect the defect. Follow the
  [test strategy](testing/TESTING_GUIDELINES.md).
- Report observed results and gaps; do not equate a scanner result or test
  count with complete security or production readiness.
- Preserve correlation IDs, safe structured logs, metrics, health probes,
  graceful shutdown, migration/recovery paths, and documented rollback.
- Follow [security](SECURITY.md), [deployment](DEPLOYMENT.md), and the existing
  branch/PR/CI workflow. Keep task-specific evidence in GitHub rather than
  duplicating completion reports in the source tree.
