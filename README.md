# Document Intelligence System 1.9

> Enterprise-grade document processing with AI-powered insights and modern cloud-native architecture

[![CI/CD Pipeline](https://github.com/aaron-seq/Roneira-AI-LLM-powered-document-intelligence-system/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/aaron-seq/Roneira-AI-LLM-powered-document-intelligence-system/actions/workflows/ci-cd.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-0db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

## Overview

A production-ready, scalable document intelligence system that combines the power of Azure AI services with advanced language models to extract, analyze, and structure information from various document types. Built with modern Python frameworks and designed for cloud deployment.

### Key Features

- **AI-Powered Processing**: Azure Document Intelligence + GPT-4 for comprehensive document analysis
- **Modern Architecture**: FastAPI, async/await, proper dependency injection
- **Production Ready**: Docker containerization, CI/CD pipelines, monitoring
- **Cloud Native**: Optimized for Railway, Render, Vercel, and other cloud platforms
- **Security First**: JWT authentication, input validation, rate limiting
- **Scalable**: Redis caching, connection pooling, async processing
- **Developer Friendly**: Comprehensive testing, type hints, documentation

## Architecture

```mermaid
graph TB
    A[Client Applications] --> B[Load Balancer]
    B --> C[FastAPI Application]
    C --> D[Document Intelligence Service]
    C --> E[Authentication Service]
    C --> F[WebSocket Manager]
    
    D --> G[Azure Document Intelligence]
    D --> H[Language Model Service]
    D --> I[Database]
    
    C --> J[Redis Cache]
    
    I --> K[(PostgreSQL/SQLite)]
    J --> L[(Redis)]
    
    style C fill:#e1f5fe
    style D fill:#f3e5f5
    style I fill:#e8f5e8
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Azure Account with AI Services
- Git

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/aaron-seq/Roneira-AI-LLM-powered-document-intelligence-system.git
   cd Roneira-AI-LLM-powered-document-intelligence-system
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Start services with Docker**
   ```bash
   docker-compose up -d redis postgres
   ```

5. **Run the application**
   ```bash
   python -m uvicorn app.main:app --reload
   ```

6. **Access the API**
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/api/docs
   - Health Check: http://localhost:8000/health

### Production Deployment

#### Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/your-template-id)

1. Connect your GitHub repository to Railway
2. Set environment variables in Railway dashboard
3. Deploy automatically with git push

#### Deploy to Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/aaron-seq/Roneira-AI-LLM-powered-document-intelligence-system)

1. Fork this repository
2. Connect to Render
3. Configure using `deployment/render.yaml`

#### Deploy to Vercel (Serverless)

```bash
npm i -g vercel
vercel --prod
```

## API Documentation

### Authentication

```bash
# Get access token
curl -X POST "http://localhost:8000/api/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=demo&password=demo"
```

### Document Upload

```bash
# Upload document
curl -X POST "http://localhost:8000/api/documents/upload" \
  -H "Authorization: Bearer <token>" \
  -F "file=@document.pdf"
```

### Check Processing Status

```bash
# Get document status
curl -X GET "http://localhost:8000/api/documents/{document_id}/status" \
  -H "Authorization: Bearer <token>"
```

### Real-time Updates

```javascript
// WebSocket connection for real-time updates
const ws = new WebSocket('ws://localhost:8000/ws/{document_id}');

ws.onmessage = function(event) {
    const update = JSON.parse(event.data);
    console.log('Processing update:', update);
};
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ENVIRONMENT` | Runtime environment | `development` | No |
| `SECRET_KEY` | JWT secret key | - | Yes |
| `DATABASE_URL` | Database connection URL | `sqlite:///./documents.db` | No |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` | No |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | - | Yes |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint | - | Yes |
| `AZURE_DOCUMENT_INTELLIGENCE_KEY` | Azure Document Intelligence key | - | Yes |
| `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` | Azure Document Intelligence endpoint | - | Yes |
| `MAX_FILE_SIZE_MB` | Maximum file size in MB | `10` | No |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | API rate limit | `60` | No |

### Example Configuration

```env
# .env file
ENVIRONMENT=production
SECRET_KEY=your-super-secure-secret-key-here
DATABASE_URL=postgresql://user:password@host:port/database
REDIS_URL=redis://host:port/0

# Azure AI Services
AZURE_OPENAI_API_KEY=your-azure-openai-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=your-document-intelligence-key
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/

# Application Settings
MAX_FILE_SIZE_MB=50
RATE_LIMIT_REQUESTS_PER_MINUTE=100
```

## Development

### Project Structure

```
┌── app/
│   ├── core/              # Core functionality
│   │   ├── authentication.py
│   │   ├── database_manager.py
│   │   └── exceptions.py
│   ├── services/          # Business logic
│   │   ├── document_intelligence_service.py
│   │   ├── cache_service.py
│   │   └── language_model_service.py
│   └── main.py            # FastAPI application
├── tests/              # Test suite
├── deployment/         # Deployment configurations
├── .github/workflows/  # CI/CD pipelines
└── config.py           # Application configuration
```

### Code Quality

This project follows modern Python development practices:

```bash
# Format code
black app/ tests/
isort app/ tests/

# Lint code
flake8 app/ tests/
mypy app/

# Security check
bandit -r app/
safety check

# Run tests
pytest tests/ -v --cov=app
```

### Adding New Features

1. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Implement feature with tests**
   - Add business logic in `app/services/`
   - Add tests in `tests/`
   - Update documentation

3. **Ensure code quality**
   ```bash
   make lint test
   ```

4. **Create pull request**
   - CI/CD pipeline will run automatically
   - Code review required for main branch

## Performance

### Benchmarks

| Metric | Value |
|--------|-------|
| Document Processing | ~5-10 seconds avg |
| Concurrent Users | 100+ supported |
| API Response Time | <200ms (health check) |
| Memory Usage | ~200MB base |
| CPU Usage | ~10% idle, ~80% processing |

### Optimization Tips

- Use Redis for caching frequently accessed data
- Implement proper connection pooling
- Monitor with application performance monitoring (APM)
- Scale horizontally with load balancers

## Security

### Security Features

- **Authentication**: JWT tokens with expiration
- **Authorization**: Role-based access control
- **Input Validation**: Comprehensive request validation
- **Rate Limiting**: Prevent API abuse
- **File Upload Security**: Type and size validation
- **Secrets Management**: Environment variable configuration
- **Security Headers**: CORS, CSP, and other security headers

### Security Best Practices

- Keep dependencies updated
- Use strong secret keys (32+ characters)
- Enable HTTPS in production
- Regular security audits with `bandit` and `safety`
- Monitor logs for suspicious activity

## Monitoring and Observability

### Health Checks

- **Application Health**: `/health` endpoint
- **Service Dependencies**: Database, Redis, Azure services
- **Performance Metrics**: Response times, error rates

### Logging

```python
# Structured logging example
logger.info(
    "Document processed",
    extra={
        "document_id": document_id,
        "user_id": user_id,
        "processing_time": processing_time,
        "status": "completed"
    }
)
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Development Environment Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Run tests before committing
pytest tests/ -v
```

## Support

- **Documentation**: [API Documentation](https://your-app.com/docs)
- **Issues**: [GitHub Issues](https://github.com/aaron-seq/Roneira-AI-LLM-powered-document-intelligence-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/aaron-seq/Roneira-AI-LLM-powered-document-intelligence-system/discussions)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

- 🎉 **Initial**: Web interface and API endpoints

---

**Made with ❤️ by [Aaron Sequeira](https://github.com/aaron-seq)**

⭐ Star this repository if you find it useful!
