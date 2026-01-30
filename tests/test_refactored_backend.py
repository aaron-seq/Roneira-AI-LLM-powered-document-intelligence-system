import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock

from backend.main import app
from backend.api.dependencies import get_document_processor, get_chat_service, get_auth_service

client = TestClient(app)

# Mocks
mock_doc_processor = AsyncMock()
mock_doc_processor.list_documents.return_value = []
mock_doc_processor.health_check.return_value = True

mock_chat_service = AsyncMock()
mock_auth_service = AsyncMock()

@pytest.fixture
def override_dependencies():
    app.dependency_overrides[get_document_processor] = lambda: mock_doc_processor
    app.dependency_overrides[get_chat_service] = lambda: mock_chat_service
    app.dependency_overrides[get_auth_service] = lambda: mock_auth_service
    yield
    app.dependency_overrides = {}

def test_root_endpoint():
    response = client.get("/api/") # Root was mounted at /api ? No.
    # main.py: app.include_router(api_router, prefix="/api")
    # But also: app.get("/", tags=["Root"])
    
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "operational"

def test_health_check():
    # Health check uses db_manager default dependency if not overridden?
    # backend/api/routers/system.py uses `get_db_manager`.
    # We didn't override it, so it might try to connect to real DB or fail.
    # It catches exception and returns "unhealthy" + 200 OK (response model).
    
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "unhealthy"]

def test_docs_endpoint():
    response = client.get("/api/docs")
    assert response.status_code == 200

def test_auth_login_validation():
    # Test missing fields
    response = client.post("/api/auth/login", data={})
    assert response.status_code == 422

# Add more tests as needed
