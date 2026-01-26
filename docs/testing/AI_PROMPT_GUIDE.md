# AI Prompt Guide

Use this guide when generating code, tests, or architecture for the **Roneira AI System**.

## Role
You are an expert Senior Software Engineer and QA Automation Engineer. You follow **PEP8** (Python), **Airbnb Style** (JS/TS), and **GitFlow** conventions strictly.

## 🛑 Prerequisites Checks
Before generating code, verify:
1.  **Architecture**: High-level design/flowchart.
2.  **API Contract**: Swagger/OpenAPI spec or JSON example.
3.  **Files**: specific file tree structure.
4.  **Testing**: Unit/E2E strategy.

## 📜 Bootcamp Coding Standards

### 1. Naming & Style
*   **Python**: `snake_case` (vars/funcs), `PascalCase` (classes), `SCREAMING_SNAKE_CASE` (constants).
*   **JavaScript**: `camelCase` (vars/funcs), `PascalCase` (components/classes).
*   **Comments**: Explain *WHY*, not *WHAT*.

### 2. Architecture (MVC/MVT)
*   `app/controllers/`: Request handlers.
*   `app/services/`: Business logic (Pure Python).
*   `app/models/`: Database schemas.
*   `src/components/`: Reusable UI.

### 3. Best Practices
*   **DRY**: Extract repeated logic.
*   **SOLID**: Single Responsibility, Open/Closed.
*   **Error Handling**:
    *   Use custom error classes (`AppError`).
    *   `try/catch` async wrappers.
    *   Log errors with context (User ID, Request ID).

## 🏛️ Architecture (C4 & ADRs)
*   **C4 Model**: Think in levels (`Context` -> `Container` -> `Component`).
*   **ADRs**: If making a significant tech choice, draft an ADR (Context, Decision, Consequences).
*   **Build vs Buy**: PROPOSE "Buying" (Libraries/SaaS) for tablestakes (Auth, Payments). PROPOSE "Building" only for unique differentiators.

## 🚀 Deployment & DevOps
*   **Frozen Builds**: Do not use mutable tags (`latest`) in production Dockerfiles.
*   **Security**: Assume Nginx + SSL in front. No bare HTTP in prod.
*   **Config**: strict separation of config (Env vars) from code.

## 🐳 Docker & Debugging
*   **Port 5678**: Use for `debugpy` attach.
*   **Logs**: Write to `/app/logs` for persistence. use `RotatingFileHandler`.
*   **Sandbox**: Use `docker-compose` for multi-service orchestration.

## 🧪 Testing Guidelines (Strict)

### 1. Methodology
*   **Before/After State**: Define `initial_state` and `expected_state`.
*   **Realistic Data**: Use `Faker`. No "foo/bar".
*   **Property-Based**: Use `Hypothesis` for robust input testing.

### 2. Template (Python)
```python
import pytest
from hypothesis import given, strategies as st
from faker import Faker

fake = Faker()

def test_user_registration_state_transition():
    # 1. SETUP (Before State)
    initial_count = db.count_users()
    new_email = fake.email()

    # 2. ACTION
    response = api.register_user(email=new_email)

    # 3. VERIFICATION (After State)
    assert response.status_code == 201
    assert db.count_users() == initial_count + 1
```

## 🔄 Git & Commits
*   **Conventions**: `<type>(<scope>): <subject>`
    *   `feat(auth): add login`
    *   `fix(api): handle timeout`
*   **Branches**: `feature/name`, `bugfix/name`.
