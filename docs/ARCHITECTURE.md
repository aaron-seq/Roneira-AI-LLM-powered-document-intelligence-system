# System Architecture

## Overview

This document outlines the architectural design of the Roneira Document Intelligence System, following the C4 Model (Context, Containers, Components).

## 1. System Context (Level 1)

High-level view of how the system interacts with users and external systems.

```mermaid
C4Context
    title System Context Diagram for Roneira AI

    Person(user, "User", "A user utilizing the Roneira system for document analysis.")
    System(roneira, "Roneira AI System", "Platform for intelligent document processing and specific insights.")

    System_Ext(ollama, "Ollama (Local LLM)", "Provides local inference capabilities.")
    System_Ext(email, "Email System", "Sends notifications and reports (optional).")

    Rel(user, roneira, "Uploads documents, views insights", "HTTPS")
    Rel(roneira, ollama, "Sends text for inference", "HTTP/JSON")
    Rel(roneira, email, "Sends emails", "SMTP")
```

## 2. Container Diagram (Level 2)

The high-level technical building blocks.

```mermaid
C4Container
    title Container Diagram for Roneira AI

    Person(user, "User", "End user")

    Container_Boundary(c1, "Roneira AI System") {
        Container(spa, "Single Page Application", "React, Vite, TypeScript", "Provides dashboard and upload UI.")
        Container(api, "API Gateway / Backend", "Python, FastAPI", "Handles requests, document processing, and business logic.")
        ContainerDb(db, "Relational Database", "SQLite/PostgreSQL", "Stores user data, metadata, and processing status.")
        ContainerDb(vector, "Vector Store", "ChromaDB/FAISS", "Stores document embeddings for semantic search.")
        Container(worker, "Background Worker", "AsyncIO/Celery", "Handles heavy document parsing and OCR tasks.")
    }

    System_Ext(ollama, "Ollama", "Local LLM Inference Engine")

    Rel(user, spa, "Uses", "HTTPS")
    Rel(spa, api, "API Calls", "JSON/HTTPS")
    Rel(api, db, "Reads/Writes", "SQL")
    Rel(api, vector, "Reads/Writes", "Vector API")
    Rel(api, ollama, "Inference Request", "HTTP")
    Rel(api, worker, "Dispatches tasks", "In-Process/Queue")
```

## 3. Component Diagram (Level 3) - Backend API

Drilling down into the FastAPI Backend structure.

```mermaid
C4Component
    title Component Diagram - Backend API

    Container(api, "API Application", "FastAPI", "Main entry point.")

    Component(auth, "Auth Controller", "FastAPI Router", "Handles login and token generation.")
    Component(doc_ctrl, "Document Controller", "FastAPI Router", "Handles document uploads and status checks.")
    Component(chat_ctrl, "Chat Controller", "FastAPI Router", "Handles RAG chat and search.")
    Component(sys_ctrl, "System Controller", "FastAPI Router", "Health checks and dashboard metrics.")

    Component(doc_service, "Document Service (Block)", "Python Class", "Orchestrates parsing, extraction, and storage.")
    Component(llm_service, "LLM Service (Block)", "Python Class", "Manages interaction with Ollama and prompts.")

    Component(parser, "PDF Parser (Helper)", "Python Module", "Extracts text from PDFs.")
    Component(telemetry, "Telemetry (Util)", "Python Module", "Structured logging and metrics.")

    Rel(api, auth, "Routes to")
    Rel(api, doc_ctrl, "Routes to")
    Rel(api, chat_ctrl, "Routes to")
    Rel(api, sys_ctrl, "Routes to")

    Rel(doc_ctrl, doc_service, "Calls")
    Rel(doc_service, parser, "Uses")
    Rel(doc_service, llm_service, "Uses")
    Rel(doc_service, telemetry, "Emits data to")
```

## 4. Code Concepts & Directory Structure

To maintain scalability and developer velocity, we strictly follow the **Block / Helpers / Utils** pattern.

### 4.1 Block (Business Logic)

**Location**: `backend/services/`

- Contains the core _domain logic_ of the application.
- Orchestrates data flow between controllers, data access, and external services.
- **Rules**:
  - Pure Python.
  - No direct HTTP/API framework dependencies (agnostic).
  - Must emit telemetry.

### 4.2 Helpers (Project Utilities)

**Location**: `backend/models/` or `backend/helpers/`

- Domain-specific utilities that are reused across blocks but are tied to _this specific project_.
- Examples: `PDFParser`, `CostCalculator`, `PermissionChecker`.

### 4.3 Utils (Shared Utilities)

**Location**: `backend/utils/`

- Generic, pure functions that could theoretically be packaged as a separate library.
- Examples: `format_date`, `generate_uuid`, `encrypt_string`.
- **Rules**:
  - No dependencies on project business logic.
  - Side-effect free where possible.

### 4.4 Structure Map

```text
backend/
├── main.py                 # Entry point
├── api/                    # Controllers (Routes)
│   └── routers/            # Route Handlers
│       ├── auth.py
│       ├── chat.py
│       └── ...
├── services/               # BLOCKS (Business Logic)
│   ├── document_service.py
│   └── local_llm_service.py
├── common/                 # HELPERS & UTILS
│   ├── helpers.py          # Project-specific helpers (Parsers)
│   └── utils.py            # Generic shared utils
├── core/                   # Config & Security
├── models/                 # Data Models (Pydantic/SQLAlchemy)
└── observability/          # Telemetry & Logging
```
