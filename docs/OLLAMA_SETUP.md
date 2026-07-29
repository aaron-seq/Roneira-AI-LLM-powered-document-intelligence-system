# Setting up Ollama

Per-platform instructions for the local LLM that powers summaries and
chat answers. **Ollama is optional** — without it the service still
extracts, indexes and searches documents; it just cannot write answers in
prose. `/api/health` reports `degraded` with the reason when it is absent.

For production deployment see [DEPLOYMENT.md](DEPLOYMENT.md); for the
full configuration reference see [`.env.example`](../.env.example).

> The Azure setup that used to live in this file has been removed: the
> current backend runs against Ollama and does not call Azure services.

---

## Quick Start (Local Development)

### Prerequisites

1. **Python 3.11+**
2. **Ollama** - Local LLM runtime
3. **Git** - Version control

### Step 1: Install Ollama

**Windows:**
```powershell
# Download and install from https://ollama.ai/download
# Or use winget:
winget install Ollama.Ollama
```

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Step 2: Pull LLM Model

```bash
# Enterprise recommended (balanced performance)
ollama pull llama3.1:8b

# Lightweight option (faster, less resource)
ollama pull llama3.2:3b

# Alternative options
ollama pull mistral:7b
ollama pull deepseek-coder:6.7b
```

### Step 3: Clone Repository

```bash
git clone https://github.com/aaron-seq/Roneira-AI-LLM-powered-document-intelligence-system.git
cd Roneira-AI-LLM-powered-document-intelligence-system
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Configure Environment

Create `.env` file in the project root:

```env
# Application Settings
ENVIRONMENT=development
SECRET_KEY=dev-secret-key-for-local-development-only-32chars
DEBUG=true

# Ollama LLM Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
OLLAMA_TIMEOUT=120

# Database (SQLite for local development)
DATABASE_URL=sqlite+aiosqlite:///./document_intelligence.db

# File Upload Settings
UPLOAD_DIRECTORY=./uploads
PROCESSED_FILES_DIRECTORY=./processed
MAX_FILE_SIZE=52428800

# Logging
LOG_LEVEL=INFO
```

### Step 6: Start the Application

```bash
# Ensure Ollama is running
ollama serve

# In another terminal, start the backend
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

### Step 7: Verify Installation

1. **Health check**: http://127.0.0.1:8000/api/health — with Ollama
   running, the `llm` component reports `ok`; without it, `degraded`.
2. **API Documentation**: http://127.0.0.1:8000/api/docs
3. **Root Endpoint**: http://127.0.0.1:8000/

---

## Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama service
ollama serve
```

### Model Not Found

```bash
# List available models
ollama list

# Pull required model
ollama pull llama3.2:3b
```

### Import Errors

```bash
# Install all dependencies
pip install -r requirements.txt

# Key dependencies
pip install sentence-transformers transformers chromadb
```

### Port Already in Use

```bash
# Find process using port 8000
netstat -ano | findstr :8000

# Use different port
python -m uvicorn backend.main:app --port 8001
```

---

## API quick reference

Every document and chat endpoint needs an `Authorization: Bearer` token from
`POST /api/auth/token`. The full list is in the
[README](../README.md#api) and at `/api/docs`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Component-level health |
| `/api/docs` | GET | Interactive API documentation |
| `/api/auth/token` | POST | Exchange credentials for a token |
| `/api/documents/upload` | POST | Upload a document |
| `/api/documents/{id}/status` | GET | Processing status |
| `/api/search` | POST | Search with citations |
| `/api/chat` | POST | Grounded question answering |
