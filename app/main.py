"""Main FastAPI application for Document Intelligence System.

Provides a modern, production-ready API with proper error handling,
validation, and deployment-friendly configuration.
"""

import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Depends,
    WebSocket,
    WebSocketDisconnect,
    UploadFile,
    BackgroundTasks,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config import application_settings
from app.core.database_manager import (
    get_database_session,
    initialize_database_connection,
)
from app.services.document_intelligence_service import DocumentIntelligenceService
from app.services.cache_service import CacheService
from app.core.authentication import create_access_token, get_authenticated_user
from app.core.websocket_manager import WebSocketConnectionManager
from app.core.exceptions import (
    DocumentProcessingError,
    ValidationError,
    ServiceUnavailableError,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, application_settings.log_level),
    format=application_settings.log_format,
)
logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Global services
websocket_manager = WebSocketConnectionManager()
document_service: Optional[DocumentIntelligenceService] = None
cache_service: Optional[CacheService] = None


# Pydantic models for API requests and responses
class HealthCheckResponse(BaseModel):
    """Health check response model."""

    status: str = Field(description="Service status")
    version: str = Field(description="Application version")
    timestamp: datetime = Field(description="Response timestamp")
    environment: str = Field(description="Current environment")
    services: Dict[str, str] = Field(description="Service status")


class AuthenticationToken(BaseModel):
    """Authentication token response."""

    access_token: str = Field(description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(description="Token expiry in seconds")


class DocumentUploadResponse(BaseModel):
    """Document upload response."""

    document_id: str = Field(description="Unique document identifier")
    filename: str = Field(description="Original filename")
    status: str = Field(description="Processing status")
    upload_timestamp: datetime = Field(description="Upload timestamp")
    estimated_processing_time: Optional[int] = Field(
        description="Estimated processing time in seconds"
    )


class DocumentStatusResponse(BaseModel):
    """Document processing status response."""

    document_id: str = Field(description="Document identifier")
    status: str = Field(description="Processing status")
    progress_percentage: int = Field(description="Processing progress")
    created_at: datetime = Field(description="Document creation timestamp")
    processed_at: Optional[datetime] = Field(
        description="Processing completion timestamp"
    )
    error_message: Optional[str] = Field(description="Error message if failed")
    extracted_data: Optional[Dict] = Field(description="Extracted document data")


class ErrorResponse(BaseModel):
    """Standard error response."""

    error_code: str = Field(description="Error code")
    error_message: str = Field(description="Human-readable error message")
    details: Optional[Dict] = Field(description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


@asynccontextmanager
async def application_lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    global document_service, cache_service

    logger.info(
        f"Starting Document Intelligence System v{application_settings.application_version}"
    )

    try:
        # Initialize upload directory
        upload_path = Path(application_settings.upload_directory)
        upload_path.mkdir(parents=True, exist_ok=True)

        # Initialize database connection
        await initialize_database_connection()
        logger.info("Database connection initialized")

        # Initialize services
        document_service = DocumentIntelligenceService()
        cache_service = CacheService()

        await document_service.initialize()
        await cache_service.initialize()

        logger.info("All services initialized successfully")

        yield

    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Shutting down services")
        if cache_service:
            await cache_service.cleanup()
        if document_service:
            await document_service.cleanup()


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Document Intelligence API",
        description="Enterprise-grade document processing with AI-powered insights",
        version=application_settings.application_version,
        lifespan=application_lifespan,
        docs_url="/api/docs" if not application_settings.is_production else None,
        redoc_url="/api/redoc" if not application_settings.is_production else None,
    )

    # Add middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=application_settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if application_settings.is_production:
        app.add_middleware(
            TrustedHostMiddleware, allowed_hosts=application_settings.allowed_origins
        )

    # Add rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    return app


app = create_application()


# Exception handlers
@app.exception_handler(DocumentProcessingError)
async def document_processing_exception_handler(request, exc: DocumentProcessingError):
    """Handle document processing errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error_code="DOCUMENT_PROCESSING_ERROR",
            error_message=str(exc),
            details={"document_id": getattr(exc, "document_id", None)},
        ).dict(),
    )


@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc: ValidationError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error_code="VALIDATION_ERROR", error_message=str(exc)
        ).dict(),
    )


# API Routes
@app.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Get the health status of the application and its services",
)
@limiter.limit("30/minute")
async def health_check(request):
    """Comprehensive health check endpoint."""
    services_status = {
        "database": "healthy",
        "cache": "healthy"
        if cache_service and await cache_service.health_check()
        else "unhealthy",
        "document_intelligence": "healthy"
        if document_service and await document_service.health_check()
        else "unhealthy",
    }

    overall_status = (
        "healthy"
        if all(status == "healthy" for status in services_status.values())
        else "degraded"
    )

    return HealthCheckResponse(
        status=overall_status,
        version=application_settings.application_version,
        timestamp=datetime.utcnow(),
        environment=application_settings.environment,
        services=services_status,
    )


@app.post(
    "/api/auth/token",
    response_model=AuthenticationToken,
    summary="Authenticate User",
    description="Authenticate user credentials and return JWT token",
)
@limiter.limit("5/minute")
async def authenticate_user(
    request,
    username: str = Form(..., description="Username"),
    password: str = Form(..., description="Password"),
):
    """Authenticate user and return JWT token."""
    # In production, implement proper user authentication
    if not username or not password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )

    access_token = create_access_token(data={"sub": username})

    return AuthenticationToken(
        access_token=access_token,
        expires_in=application_settings.token_expiry_minutes * 60,
    )


@app.post(
    "/api/documents/upload",
    response_model=DocumentUploadResponse,
    summary="Upload Document",
    description="Upload a document for AI-powered processing and analysis",
)
@limiter.limit("10/minute")
async def upload_document(
    request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file to upload"),
    user: str = Depends(get_authenticated_user),
):
    """Upload and queue document for processing."""

    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No file provided"
        )

    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in application_settings.allowed_file_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {file_extension} not allowed",
        )

    # Check file size
    content = await file.read()
    if len(content) > application_settings.max_file_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds {application_settings.max_file_size_mb}MB limit",
        )

    # Generate unique document ID
    document_id = str(uuid.uuid4())
    upload_timestamp = datetime.utcnow()

    # Save file
    file_path = (
        Path(application_settings.upload_directory) / f"{document_id}_{file.filename}"
    )
    with open(file_path, "wb") as buffer:
        buffer.write(content)

    # Queue for background processing
    background_tasks.add_task(
        process_document_background, document_id, str(file_path), user, file.filename
    )

    return DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename,
        status="queued",
        upload_timestamp=upload_timestamp,
        estimated_processing_time=30,
    )


@app.get(
    "/api/documents/{document_id}/status",
    response_model=DocumentStatusResponse,
    summary="Get Document Status",
    description="Retrieve the processing status and results of a document",
)
async def get_document_status(
    document_id: str, user: str = Depends(get_authenticated_user)
):
    """Get document processing status and results."""

    if not cache_service:
        raise ServiceUnavailableError("Cache service unavailable")

    status_data = await cache_service.get_document_status(document_id)
    if not status_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )

    return DocumentStatusResponse(**status_data)


@app.websocket("/ws/{document_id}")
async def websocket_document_updates(websocket: WebSocket, document_id: str):
    """WebSocket endpoint for real-time document processing updates."""
    await websocket_manager.connect(websocket, document_id)

    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        await websocket_manager.disconnect(websocket, document_id)
    except Exception as e:
        logger.error(f"WebSocket error for document {document_id}: {e}")
        await websocket_manager.disconnect(websocket, document_id)


@app.get(
    "/api/documents",
    summary="List Documents",
    description="Get a list of processed documents for the authenticated user",
)
async def list_user_documents(
    user: str = Depends(get_authenticated_user), limit: int = 10, offset: int = 0
):
    """List user's documents with pagination."""

    if not document_service:
        raise ServiceUnavailableError("Document service unavailable")

    documents = await document_service.get_user_documents(
        user_id=user, limit=limit, offset=offset
    )

    return {
        "documents": documents,
        "total": len(documents),
        "limit": limit,
        "offset": offset,
    }


# Background task for document processing
async def process_document_background(
    document_id: str, file_path: str, user_id: str, original_filename: str
):
    """Background task to process uploaded documents."""

    if not document_service or not cache_service:
        logger.error("Required services not available")
        return

    try:
        # Update status to processing
        await cache_service.update_document_status(
            document_id,
            {
                "status": "processing",
                "progress_percentage": 10,
                "created_at": datetime.utcnow(),
            },
        )

        await websocket_manager.broadcast_to_document(
            document_id,
            {"document_id": document_id, "status": "processing", "progress": 10},
        )

        # Process document
        processing_result = await document_service.process_document(
            document_id=document_id,
            file_path=file_path,
            user_id=user_id,
            original_filename=original_filename,
        )

        # Update final status
        final_status = {
            "document_id": document_id,
            "status": "completed" if processing_result.get("success") else "failed",
            "progress_percentage": 100,
            "created_at": datetime.utcnow(),
            "processed_at": datetime.utcnow(),
            "extracted_data": processing_result.get("data"),
            "error_message": processing_result.get("error"),
        }

        await cache_service.update_document_status(document_id, final_status)

        await websocket_manager.broadcast_to_document(document_id, final_status)

    except Exception as e:
        logger.error(f"Error processing document {document_id}: {e}")

        error_status = {
            "document_id": document_id,
            "status": "failed",
            "progress_percentage": 0,
            "error_message": str(e),
            "processed_at": datetime.utcnow(),
        }

        await cache_service.update_document_status(document_id, error_status)
        await websocket_manager.broadcast_to_document(document_id, error_status)

    finally:
        # Clean up temporary file
        try:
            os.remove(file_path)
        except OSError:
            logger.warning(f"Failed to remove temporary file: {file_path}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=application_settings.server_host,
        port=application_settings.server_port,
        reload=application_settings.is_development,
        workers=1
        if application_settings.is_development
        else application_settings.worker_count,
    )
