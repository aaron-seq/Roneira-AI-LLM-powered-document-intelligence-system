# Architecture Bootcamp Standards

## 1. Core Principles
*   **Architecture = Decision + Documentation**: It's about consciously deciding what to include/exclude and documenting how components interconnect.
*   **Defect Cost**: Fixing defects in production costs **50-200x** more than fixing them during design. Phase containment is key.
*   **Cone of Uncertainty**: Estimates vary by 4x initially. Architecture work during Backlog Refinement narrows this.
*   **Timing**: Budget ~5 days for architecture/NFRs before coding a major epic.

## 2. The C4 Model (Visualization)
We use the C4 model to map architecture at different zoom levels.
*   **Level 1: System Context**: The big picture. Users <-> System <-> External Systems. (Accessible to stakeholders).
*   **Level 2: Containers**: Applications and Data Stores (e.g., React App, API, Postgres, Redis).
*   **Level 3: Components**: Internal structure of a container (e.g., AuthController, PaymentService).
*   **Level 4: Code**: Classes and interfaces (Diagrams usually generated from code).

**Tooling**: Use **Structurizr** (Diagrams as Code) over generic drawing tools.

## 3. Architecture Decision Records (ADRs)
Document the *why* behind significant decisions.
*   **Context**: What is the problem? behavior? constraints?
*   **Decision**: What did we choose?
*   **Status**: Proposed / Accepted / Deprecated.
*   **Consequences**: Trade-offs, tech debt, future impact.

## 4. Build vs Buy (Decision Framework)
*   **Tablestakes**: Essential but common features (Auth, Logging, Payments). **BUY / USE EXISTING**. Do not reinvent.
*   **Delighters**: Unique differentiators. **BUILD**. Focus engineering effort here.
*   **Evaluation**: Check Technology Radar (Adopt/Trial/Assess/Hold).

## 5. Delighter Patterns
*   **Microservices**: For independent scaling and independent teams.
*   **Cloud Native**: Auto-scaling, managed services.
*   **Event-Driven**: For real-time responsiveness and decoupling.
