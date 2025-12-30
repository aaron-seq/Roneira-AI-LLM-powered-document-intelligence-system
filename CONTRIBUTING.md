# Contributing to Document Intelligence System

We welcome contributions from the community. This guide will help you get started with contributing to the project.

## Development Setup

### Prerequisites

- Python 3.11 or higher
- Docker and Docker Compose
- Git
- Virtual environment tool (venv or conda)

### Initial Setup

1. Fork the repository
2. Clone your fork locally
   ```bash
   git clone https://github.com/YOUR_USERNAME/Roneira-AI-LLM-powered-document-intelligence-system.git
   cd Roneira-AI-LLM-powered-document-intelligence-system
   ```

3. Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

5. Set up environment variables
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Running Tests

### All Tests
```bash
pytest tests/ -v --cov=backend --cov-report=html
```

### RAG Service Tests
The project includes 24 comprehensive unit tests for RAG services:
```bash
pytest tests/test_rag_services.py -v
```

### WebSocket Tests
```bash
pytest tests/test_websocket.py -v
```

### Specific Test
```bash
pytest tests/test_rag_services.py::TestEmbeddingService::test_embed_query -v
```

## Code Quality Standards

We maintain high code quality standards. Before submitting a pull request, ensure your code passes all quality checks.

### Formatting
```bash
# Format code with black
black backend/ tests/

# Sort imports
isort backend/ tests/
```

### Linting
```bash
# Check code quality
flake8 backend/ tests/ --max-line-length=100

# Type checking
mypy backend/
```

### Security Scanning
```bash
# Check for security issues
bandit -r backend/

# Check dependencies
safety check
```

### All Quality Checks
```bash
# Run all checks at once
make lint  # If Makefile is configured
# or
black . && isort . && flake8 && mypy backend/ && bandit -r backend/
```

## Pull Request Process

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

- Write clean, readable code
- Follow existing code style
- Add docstrings to functions and classes
- Include type hints where appropriate

### 3. Add Tests

- Write unit tests for new functionality
- Ensure tests cover edge cases
- Maintain or improve test coverage (target: 80%+)

### 4. Update Documentation

- Update README.md if adding features
- Add docstrings to new functions/classes
- Update API documentation if endpoints change

### 5. Commit Your Changes
```bash
git add .
git commit -m "feat: add semantic search capability"
# or
git commit -m "fix: resolve vector store connection issue"
```

Commit message format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions or modifications
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

### 6. Push and Create PR
```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear title and description
- Reference to related issues
- Screenshots/examples if applicable

### 7. Code Review

- Address review feedback promptly
- Keep the conversation professional
- Update PR based on suggestions

## Development Guidelines

### RAG Development

When working on RAG (Retrieval-Augmented Generation) features:

1. **Embedding Models**: Document any changes to embedding models or dimensions
2. **Vector Stores**: Test with both ChromaDB and in-memory fallback
3. **Chunking Strategies**: Validate text splitting maintains context
4. **Retrieval Quality**: Measure and document retrieval accuracy
5. **Performance**: Monitor embedding generation and search times

### Adding New Services

When adding new services to `backend/services/`:

1. Follow the existing service pattern
2. Add comprehensive error handling
3. Include logging for debugging
4. Write unit tests in `tests/test_*.py`
5. Update service exports in `__init__.py`

### API Endpoints

When adding new API endpoints:

1. Use appropriate HTTP methods (GET, POST, PUT, DELETE)
2. Include request/response models
3. Add input validation
4. Document in README.md
5. Include example curl commands

### Database Changes

When modifying database models:

1. Create migration scripts if using Alembic
2. Test with both SQLite and PostgreSQL
3. Document schema changes
4. Update test fixtures

## Testing Best Practices

### Unit Tests

- Test individual functions in isolation
- Use mocks for external dependencies
- Follow AAA pattern (Arrange, Act, Assert)
- Name tests descriptively: `test_embedding_service_handles_empty_input`

### Integration Tests

- Test component interactions
- Use test databases
- Clean up test data after tests

### Example Test Structure
```python
def test_vector_store_similarity_search():
    """Test that similarity search returns relevant documents."""
    # Arrange
    vector_store = VectorStoreService()
    documents = ["test doc 1", "test doc 2"]
    
    # Act
    vector_store.add_documents(documents)
    results = vector_store.search("test query", k=2)
    
    # Assert
    assert len(results) == 2
    assert all(isinstance(r, Document) for r in results)
```

## Areas We Need Help

Contributions are especially welcome in these areas:

### High Priority
- Enhancing retrieval algorithms for better accuracy
- Adding support for additional embedding models
- Improving guardrail detection for edge cases
- Performance optimization for large document sets
- Additional document format support

### Medium Priority
- Expanding test coverage
- Documentation improvements
- Example applications and tutorials
- Integration with additional LLM providers

### Nice to Have
- UI/UX improvements for the dashboard
- Additional deployment configurations
- Monitoring and observability enhancements
- Internationalization support

## Project Structure

```
Roneira-AI-LLM-powered-document-intelligence-system/
├── backend/
│   ├── services/              # Business logic and RAG services
│   │   ├── text_splitter_service.py
│   │   ├── embedding_service.py
│   │   ├── vector_store_service.py
│   │   ├── retrieval_service.py
│   │   ├── memory_service.py
│   │   ├── chat_service.py
│   │   ├── guardrail_service.py
│   │   └── prompt_service.py
│   ├── main.py               # FastAPI application
│   └── config.py             # Configuration management
├── tests/                     # Test suite
│   ├── test_rag_services.py  # RAG unit tests (24 tests)
│   └── test_websocket.py     # WebSocket tests
├── deployment/               # Deployment configurations
├── frontend/                 # Frontend application
└── requirements.txt          # Python dependencies
```

## Getting Help

- GitHub Issues: Report bugs or request features
- GitHub Discussions: Ask questions and share ideas
- Pull Requests: Submit code changes

## Code of Conduct

Please note that this project adheres to professional standards:

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help create a positive community

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be recognized in the project README. Significant contributions may result in collaborator access.

Thank you for contributing to the Document Intelligence System!
