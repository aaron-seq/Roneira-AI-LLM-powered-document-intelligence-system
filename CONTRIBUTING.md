# Contributing to Document Intelligence System

We welcome contributions! This guide enforces our **Strict Architecture & Code Quality Standards**.

> **"Architecture = Decision + Documentation"**

## 1. Strictly Enforced Architecture

### 1.1 Code Organization (Block / Helpers / Utils)
To maintain scalability, code MUST be organized into these three layers:
*   **BLOCK (Business Logic)**: The core domain logic. Pure Python/TS. No UI code. No raw database calls (use repositories/models).
    *   *Location*: `backend/services/` or `src/services/`
*   **HELPERS (Project Utilities)**: Logic specific to *this* project but reusable across blocks.
    *   *Location*: `backend/helpers/` or `src/utils/project/`
*   **UTILS (Shared Utilities)**: Generic, pure functions (Date formatting, String manipulation). Could be extracted to a separate NPM/PyPI package.
    *   *Location*: `backend/utils/` or `src/utils/generic/`

### 1.2 Separation of Concerns
*   **UI != Data Fetching**: UI components must NEVER fetch data directly. They must call an **SDK** or **Service Layer**.
*   **Dependencies Flow Down**: UI -> Block -> Data. Never the reverse.

### 1.3 SDKs & Microservices
*   **SDKs Enable Scale**: If you have a reusable pattern or an internal service, create an SDK wrapper for it. Do not duplicate API call logic.

---

## 2. Production Reliability Standards

### 2.1 Telemetry & Observability
*   **Telemetry != Analytics**:
    *   **Telemetry**: System health (Latency, Error Rates, CPU). Implementation: **OpenTelemetry**.
    *   **Analytics**: Business value (User Registrations, Document Counts). Implementation: **PostHog/Mixpanel**.
*   **Mandate**: Every new Service/Block must emit **latency** metrics and **error** counts.

### 2.2 Critical Error Handling
*   **No Silent Failures**: NEVER use `except: pass` or empty `catch` blocks.
*   **Full Context**: Logs must include stack traces and correlation IDs.
*   **Pattern**:
    ```python
    try:
        # Block Logic
    except SpecificError as e:
        logger.error("Failed to process document", error=e, doc_id=id, stack_info=True)
        raise AppError("Processing Failed") from e
    ```

---

## 3. Contribution Checklists

Before opening a PR, you MUST check all boxes.

### ✅ Code Quality Checklist
- [ ] **Organization**: Logic separated into Block, Helpers, Utils.
- [ ] **Comments**: Explain "WHY", not just "WHAT".
- [ ] **AI-Friendly**: Complex logic has comments aiding LLM understanding.
- [ ] **Sunny & Rainy**: Tests cover Happy Path AND Failure Scenarios.
- [ ] **Variables**: Descriptive naming (no `x`, `data`, `item`).

### ✅ Architecture Checklist
- [ ] **UI Agnostic**: Components receive data via props/hooks, not internal fetch.
- [ ] **Layering**: UI -> BLoC -> Data flow respected.
- [ ] **SOLID**: Single Responsibility Pattern applied.

### ✅ Reliability Checklist
- [ ] **Telemetry**: Request latency and error rates tracking added.
- [ ] **Logging**: Full stack traces on errors.
- [ ] **Environment**: No hardcoded secrets; use strict Env Vars.

---

## 4. Development Setup

### Python Backend
```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Test (Deep Audit)
pytest tests/ -v
```

### Git Workflow
*   **Branching**: `feature/desc`, `fix/desc`.
*   **Commits**: Conventional Commits (`feat(auth): add login`).
*   **PRs**: Must include "Before/After" state verification proof.
