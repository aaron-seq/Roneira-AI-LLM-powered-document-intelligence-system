# Architecture Bootcamp Standards

Use [ARCHITECTURE.md](../ARCHITECTURE.md) for the implemented system and
[CONTRIBUTING.md](../../CONTRIBUTING.md#project-conventions) for code boundaries.

## C4 views

Document only the level needed to explain the change: system context
(users/external systems), containers (applications/stores), components
(responsibilities/dependencies), or code. Reuse existing diagrams and tools.

## Architecture decisions

For a durable trade-off, record context, decision, status, consequences, and
the evidence/alternatives considered. Keep routine choices in the issue or PR.
Reuse maintained solutions for commodity capabilities and justify custom
domain code, service boundaries, and scaling from real requirements.

Security, testing, operational, and agent expectations are maintained in the
[engineering reference](../AI_BOOTCAMP_CONTEXT_MASTER.md), rather than repeated here.
