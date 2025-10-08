"""
Enhanced FastAPI Application for Document Intelligence System
Implements modern async patterns, proper error handling, and performance optimization
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional
import uuid

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
    Request
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
import uvicorn

from backend.core.config import get_application_settings
from backend.core.database import DatabaseManager
from backend.services.document_processor import DocumentProcessorService
from backend.services.websocket_manager import WebSocketManager
from backend.services.auth_service import AuthService
from backend.models.responses import (
    HealthCheckResponse,
    DocumentUploadResponse,
    DocumentStatusResponse,
    AuthTokenResponse
)
from backend.core.exceptions import DocumentProcessingError
from backend.core.middleware import add_security_headers

# Configure logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

@asynccontextmanager
async def application_lifespan(app: FastAPI):
    """Enhanced application lifecycle management"""
    settings = get_application_settings()
    logger.info("🚀 Starting Document Intelligence System...")
    
    # Initialize core services
    db_manager = DatabaseManager()
    await db_manager.initialize()
    
    websocket_manager = WebSocketManager()
    document_processor = DocumentProcessorService()
    auth_service = AuthService()
    
    # Store in app state for access in endpoints
    app.state.db_manager = db_manager
    app.state.websocket_manager = websocket_manager
    app.state.document_processor = document_processor
    app.state.auth_service = auth_service
    
    # Ensure upload directories exist
    os.makedirs(settings.upload_directory, exist_ok=True)
    os.makedirs(settings.processed_files_directory, exist_ok=True)
    
    logger.info("✅ System initialization complete")
    yield
    
    logger.info("🛑 Shutting down gracefully...")
    await db_manager.close()

def create_application() -> FastAPI:
    """Factory function to create FastAPI application with proper configuration"""
    settings = get_application_settings()
    
    application = FastAPI(
        title="🤖 AI Document Intelligence System",
        description="Enterprise-grade document processing with AI/ML capabilities",
        version="2.0.0",
        docs_url="/api/docs" if settings.debug else None,
        redoc_url="/api/redoc" if settings.debug else None,
        lifespan=application_lifespan,
    )
    
    # Security middleware
    application.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts
    )
    
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    # Custom security headers middleware
    application.middleware("http")(add_security_headers)
    
    return application

app = create_application()

# API Routes
@app.get("/health", response_model=HealthCheckResponse, tags=["System"])
async def health_check() -> HealthCheckResponse:
    """Enhanced health check with system status"""
    return HealthCheckResponse(
        status="healthy",
        version="2.0.0",
        timestamp=datetime.utcnow(),
        database_status="connected",
        services_status="operational"
    )

@app.post("/api/auth/login", response_model=AuthTokenResponse, tags=["Authentication"])
async def authenticate_user(
    request: Request,
    username: str = Form(...),
    password: str = Form(...)
) -> AuthTokenResponse:
    """Enhanced authentication with rate limiting"""
    auth_service: AuthService = request.app.state.auth_service
    
    try:
        token_data = await auth_service.authenticate_user(username, password)
        return AuthTokenResponse(**token_data)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )

@app.post("/api/documents/upload", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_document_for_processing(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file to process"),
    extract_tables: bool = Form(default=True),
    extract_images: bool = Form(default=False),
    current_user: dict = Depends(lambda: {"id": "demo_user"})  # Simplified for demo
) -> DocumentUploadResponse:
    """Enhanced document upload with validation and processing options"""
    
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )
    
    # Validate file type and size
    allowed_types = {'.pdf', '.docx', '.doc', '.txt', '.png', '.jpg', '.jpeg'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file_extension}"
        )
    
    # Generate unique document ID
    document_id = str(uuid.uuid4())
    
    try:
        document_processor: DocumentProcessorService = request.app.state.document_processor
        websocket_manager: WebSocketManager = request.app.state.websocket_manager
        
        # Save file securely
        file_path = await document_processor.save_uploaded_file(
            file, document_id, current_user["id"]
        )
        
        # Queue for background processing
        background_tasks.add_task(
            process_document_async,
            request.app.state,
            document_id,
            file_path,
            {
                "extract_tables": extract_tables,
                "extract_images": extract_images,
                "user_id": current_user["id"]
            }
        )
        
        return DocumentUploadResponse(
            document_id=document_id,
            status="queued",
            message="Document uploaded successfully and queued for processing",
            filename=file.filename,
            upload_timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Upload error for user {current_user['id']}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document upload failed. Please try again."
        )

@app.get("/api/documents/{document_id}/status", response_model=DocumentStatusResponse, tags=["Documents"])
async def get_document_processing_status(
    document_id: str,
    request: Request
) -> DocumentStatusResponse:
    """Get real-time document processing status"""
    try:
        document_processor: DocumentProcessorService = request.app.state.document_processor
        status_data = await document_processor.get_document_status(document_id)
        
        if not status_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
            
        return DocumentStatusResponse(**status_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status retrieval error for document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to retrieve document status"
        )

@app.websocket("/ws/documents/{client_id}")
async def websocket_document_updates(websocket: WebSocket, client_id: str):
    """Enhanced WebSocket connection for real-time updates"""
    websocket_manager: WebSocketManager = websocket.app.state.websocket_manager
    
    await websocket_manager.connect(websocket, client_id)
    
    try:
        while True:
            # Keep connection alive and handle incoming messages
            message = await websocket.receive_text()
            await websocket_manager.handle_client_message(client_id, message)
            
    except WebSocketDisconnect:
        await websocket_manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        await websocket_manager.disconnect(client_id)

async def process_document_async(app_state, document_id: str, file_path: str, options: dict):
    """Enhanced async document processing with detailed progress tracking"""
    document_processor: DocumentProcessorService = app_state.document_processor
    websocket_manager: WebSocketManager = app_state.websocket_manager
    
    try:
        await document_processor.process_document_with_ai(
            document_id, 
            file_path, 
            options,
            progress_callback=lambda progress: asyncio.create_task(
                websocket_manager.broadcast_progress(document_id, progress)
            )
        )
        
    except Exception as e:
        logger.error(f"Document processing failed for {document_id}: {e}")
        await websocket_manager.broadcast_error(document_id, str(e))

# Global exception handler
@app.exception_handler(DocumentProcessingError)
async def document_processing_exception_handler(request: Request, exc: DocumentProcessingError):
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.message,
            "error_code": exc.error_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
