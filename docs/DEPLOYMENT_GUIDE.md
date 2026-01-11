# Deployment Guide

## Document Intelligence System - Setup & Deployment

This guide covers step-by-step instructions for running the system locally with Ollama LLM and deploying to production with Azure services.

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

1. **Health Check**: http://127.0.0.1:8000/health
2. **API Documentation**: http://127.0.0.1:8000/api/docs
3. **Root Endpoint**: http://127.0.0.1:8000/

---

## Azure Services Configuration

For enterprise production deployment with Azure services.

### Required Azure Services

| Service | Purpose | Required |
|---------|---------|----------|
| Azure OpenAI | LLM for chat/analysis | Optional (can use Ollama) |
| Azure Document Intelligence | OCR/document extraction | Optional (free OCR available) |
| Azure Blob Storage | File storage | Optional |
| Azure Redis Cache | Caching | Optional (in-memory fallback) |

### Azure OpenAI Setup

1. **Create Azure OpenAI Resource** in Azure Portal
2. **Deploy a model** (e.g., `gpt-4-turbo-preview`)
3. **Get API key and endpoint**

```env
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your-azure-openai-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4-turbo-preview
AZURE_OPENAI_API_VERSION=2024-02-01
```

### Azure Document Intelligence Setup

1. **Create Azure AI Document Intelligence resource**
2. **Get API key and endpoint**

```env
# Azure Document Intelligence Configuration
AZURE_DOCUMENT_INTELLIGENCE_KEY=your-azure-doc-intelligence-key
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
```

---

## Docker Deployment

### Build Docker Image

```bash
docker build -t document-intelligence:latest .
```

### Run Container

```bash
docker run -d \
  -p 8000:8000 \
  -e ENVIRONMENT=production \
  -e SECRET_KEY=your-production-secret-key \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  document-intelligence:latest
```

### Docker Compose (Full Stack)

```yaml
version: '3.8'
services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - OLLAMA_BASE_URL=http://ollama:11434
    depends_on:
      - ollama
      - redis

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

volumes:
  ollama_data:
```

---

## Production Deployment Checklist

### Security
- [ ] Set strong `SECRET_KEY` (32+ characters)
- [ ] Set `ENVIRONMENT=production`
- [ ] Set `DEBUG=false`
- [ ] Configure HTTPS/TLS
- [ ] Set up proper CORS origins

### Performance
- [ ] Use Redis for caching
- [ ] Configure PostgreSQL for database
- [ ] Set appropriate worker count

### Monitoring
- [ ] Set up health check monitoring
- [ ] Configure application logging
- [ ] Set up error alerting

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

## API Quick Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/docs` | GET | Swagger documentation |
| `/api/chat` | POST | Chat with LLM |
| `/api/documents/upload` | POST | Upload document |
| `/api/documents/{id}/status` | GET | Get document status |
| `/api/search` | POST | Semantic search |
