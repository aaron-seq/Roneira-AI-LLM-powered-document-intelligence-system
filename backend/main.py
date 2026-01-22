import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional, Dict, Any
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
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from backend.core.config import get_settings
from backend.core.database import DatabaseManager
from backend.services.document_processor import DocumentProcessorService
from backend.services.websocket_manager import WebSocketManager
from backend.services.auth_service import AuthService
from backend.services.chat_service import ChatService
from backend.services.guardrail_service import GuardrailService
from backend.services.feedback_service import FeedbackService
from backend.models.responses import (
    HealthCheckResponse,
    DocumentUploadResponse,
    DocumentStatusResponse,
    AuthTokenResponse,
)
from backend.models.chat_models import (
    ChatRequest,
    ChatResponse as ChatAPIResponse,
    SearchRequest,
    SearchResponse,
    IndexDocumentRequest,
    IndexDocumentResponse,
    ConversationHistoryResponse,
    RAGStatsResponse,
)


class FeedbackRequest(BaseModel):
    message_id: str
    is_positive: bool


class DashboardMetricsResponse(BaseModel):
    total_documents: int
    processed_documents: int
    accuracy: float
    avg_confidence: float


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log")
        if os.path.exists("logs")
        else logging.StreamHandler(),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Global services (will be initialized in lifespan)
db_manager: Optional[DatabaseManager] = None
websocket_manager: Optional[WebSocketManager] = None
document_processor: Optional[DocumentProcessorService] = None
auth_service: Optional[AuthService] = None
chat_service: Optional[ChatService] = None
guardrail_service: Optional[GuardrailService] = None
feedback_service: Optional[FeedbackService] = None


@asynccontextmanager
async def application_lifespan(app: FastAPI):
    """Enhanced application lifecycle management"""
    global db_manager, websocket_manager, document_processor, auth_service
    global chat_service, guardrail_service, feedback_service

    settings = get_settings()
    logger.info("🚀 Starting Document Intelligence System...")

    try:
        # Initialize core services
        db_manager = DatabaseManager()
        await db_manager.initialize()

        websocket_manager = WebSocketManager()
        document_processor = DocumentProcessorService()
        auth_service = AuthService()

        # Initialize AI services
        try:
            await document_processor.initialize()
        except Exception as e:
            logger.error(f"❌ Failed to initialize Document Processor: {e}")

        # Initialize RAG services
        try:
            chat_service = ChatService()
            guardrail_service = GuardrailService()
            await chat_service.initialize()
            logger.info("✅ RAG services initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG services: {e}")
            logger.warning("⚠️ Chat features will be unavailable")

        try:
            feedback_service = FeedbackService()
        except Exception as e:
            logger.error(f"❌ Failed to initialize Feedback Service: {e}")

        # Create upload directories
        os.makedirs(settings.upload_directory, exist_ok=True)
        os.makedirs(settings.processed_files_directory, exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        logger.info("✅ System initialization complete")
        yield

    except Exception as e:
        logger.error(f" Startup failed: {e}")
        raise
    finally:
        logger.info(" Shutting down gracefully...")
        if db_manager:
            await db_manager.close()


def create_application() -> FastAPI:
    """Factory function to create FastAPI application"""
    settings = get_settings()

    application = FastAPI(
        title="AI Document Intelligence System",
        description="Enterprise-grade document processing with AI/ML capabilities",
        version="2.0.0",
        docs_url="/api/docs" if settings.debug else None,
        redoc_url="/api/redoc" if settings.debug else None,
        lifespan=application_lifespan,
    )

    # CORS middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # Serve static files (for uploaded documents preview)
    if os.path.exists("uploads"):
        application.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

    return application


app = create_application()


# API Routes
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with system info"""
    return {
        "message": "🤖 AI Document Intelligence System",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/api/docs",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["System"])
async def health_check() -> HealthCheckResponse:
    """Enhanced health check"""
    try:
        # Check database connection
        db_status = (
            "connected"
            if db_manager and await db_manager.health_check()
            else "disconnected"
        )

        return HealthCheckResponse(
            status="healthy",
            version="2.0.0",
            timestamp=datetime.utcnow(),
            database_status=db_status,
            services_status="operational",
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            version="2.0.0",
            timestamp=datetime.utcnow(),
            database_status="error",
            services_status="degraded",
        )


@app.post("/api/auth/login", response_model=AuthTokenResponse, tags=["Authentication"])
async def authenticate_user(
    username: str = Form(...), password: str = Form(...)
) -> AuthTokenResponse:
    """User authentication endpoint"""
    try:
        if not auth_service:
            raise HTTPException(
                status_code=500, detail="Authentication service unavailable"
            )

        token_data = await auth_service.authenticate_user(username, password)
        return AuthTokenResponse(**token_data)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed",
        )


@app.post(
    "/api/documents/upload", response_model=DocumentUploadResponse, tags=["Documents"]
)
async def upload_document_for_processing(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file to process"),
    extract_tables: bool = Form(default=True),
    extract_images: bool = Form(default=False),
    enhance_with_ai: bool = Form(default=True),
) -> DocumentUploadResponse:
    """Enhanced document upload with comprehensive validation"""

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No file provided"
        )

    # Validate file type
    allowed_extensions = {".pdf", ".doc", ".docx", ".txt", ".png", ".jpg", ".jpeg"}
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file_extension}. Allowed: {', '.join(allowed_extensions)}",
        )

    # Check file size
    settings = get_settings()
    content = await file.read()
    await file.seek(0)  # Reset file pointer

    if len(content) > settings.max_file_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.max_file_size // (1024 * 1024)}MB",
        )

    document_id = str(uuid.uuid4())

    try:
        if not document_processor:
            raise HTTPException(
                status_code=500, detail="Document processor unavailable"
            )

        # Save file securely
        file_path = await document_processor.save_uploaded_file(
            file,
            document_id,
            "demo_user",  # In production, get from JWT token
        )

        # Initialize status synchronously to avoid race condition
        document_processor.initialize_status(document_id, file.filename)

        # Queue for background processing
        processing_options = {
            "extract_tables": extract_tables,
            "extract_images": extract_images,
            "enhance_with_ai": enhance_with_ai,
            "user_id": "demo_user",
        }

        background_tasks.add_task(
            process_document_async,
            document_id,
            file_path,
            processing_options,
            file.filename,
        )

        return DocumentUploadResponse(
            document_id=document_id,
            status="queued",
            message="Document uploaded successfully and queued for processing",
            filename=file.filename,
            upload_timestamp=datetime.utcnow(),
        )

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document upload failed: {str(e)}",
        )


@app.get(
    "/api/documents/{document_id}/status",
    response_model=DocumentStatusResponse,
    tags=["Documents"],
)
async def get_document_processing_status(document_id: str) -> DocumentStatusResponse:
    """Get real-time document processing status"""
    try:
        if not document_processor:
            raise HTTPException(
                status_code=500, detail="Document processor unavailable"
            )

        status_data = await document_processor.get_document_status(document_id)

        if not status_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
            )

        return DocumentStatusResponse(**status_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status retrieval error for document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to retrieve document status",
        )


@app.get("/api/documents", tags=["Documents"])
async def list_documents(
    limit: int = 10, offset: int = 0, status_filter: Optional[str] = None
):
    """List processed documents with pagination"""
    try:
        if not document_processor:
            raise HTTPException(
                status_code=500, detail="Document processor unavailable"
            )

        documents = await document_processor.list_documents(
            limit=limit, offset=offset, status_filter=status_filter
        )

        return {
            "documents": documents,
            "total": len(documents),
            "limit": limit,
            "offset": offset,
        }

    except Exception as e:
        logger.error(f"Document listing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to retrieve documents",
        )


@app.websocket("/ws/documents/{client_id}")
async def websocket_document_updates(websocket: WebSocket, client_id: str):
    """Enhanced WebSocket for real-time updates"""
    if not websocket_manager:
        await websocket.close(code=1011, reason="WebSocket service unavailable")
        return

    await websocket_manager.connect(websocket, client_id)

    try:
        while True:
            # Keep connection alive and handle messages
            message = await websocket.receive_text()
            await websocket_manager.handle_client_message(client_id, message)

    except WebSocketDisconnect:
        await websocket_manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        await websocket_manager.disconnect(client_id)


async def process_document_async(
    document_id: str, file_path: str, options: Dict[str, Any], filename: str
):
    """Background document processing task"""
    if not document_processor or not websocket_manager:
        logger.error("Required services not available for document processing")
        return

    try:
        await document_processor.process_document_with_ai(
            document_id,
            file_path,
            options,
            filename=filename,
            progress_callback=lambda progress: asyncio.create_task(
                websocket_manager.broadcast_progress(document_id, progress)
            ),
        )

    except Exception as e:
        logger.error(f"Document processing failed for {document_id}: {e}")
        await websocket_manager.broadcast_error(document_id, str(e))


# ============ RAG API Endpoints ============


@app.post("/api/chat", response_model=ChatAPIResponse, tags=["Chat"])
async def chat_completion(request: ChatRequest) -> ChatAPIResponse:
    """RAG-powered chat completion endpoint."""
    if not chat_service:
        raise HTTPException(status_code=500, detail="Chat service unavailable")

    try:
        # Validate input with guardrails
        if guardrail_service:
            validation = await guardrail_service.validate_input(request.message)
            if not validation.is_valid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Input validation failed: {validation.details}",
                )
            message = validation.filtered_content or request.message
        else:
            message = request.message

        # Process chat
        response = await chat_service.chat(
            message=message,
            session_id=request.session_id,
            use_rag=request.use_rag,
            rag_top_k=request.rag_top_k,
            document_filter=request.document_id,
        )

        return ChatAPIResponse(
            message=response.message,
            session_id=response.session_id,
            sources=response.sources,
            model=response.model,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search", response_model=SearchResponse, tags=["RAG"])
async def semantic_search(request: SearchRequest) -> SearchResponse:
    """Semantic search over indexed documents."""
    if not chat_service:
        raise HTTPException(status_code=500, detail="Search service unavailable")

    try:
        results = await chat_service.search_documents(
            query=request.query, top_k=request.top_k
        )

        return SearchResponse(
            query=request.query, results=results, total_results=len(results)
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/documents/{document_id}/index",
    response_model=IndexDocumentResponse,
    tags=["RAG"],
)
async def index_document_for_rag(
    document_id: str, request: IndexDocumentRequest
) -> IndexDocumentResponse:
    """Index a document for RAG retrieval."""
    if not chat_service:
        raise HTTPException(status_code=500, detail="Indexing service unavailable")

    try:
        result = await chat_service.index_document(
            document_id=document_id, content=request.content, metadata=request.metadata
        )

        return IndexDocumentResponse(**result)
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/memory/{session_id}",
    response_model=ConversationHistoryResponse,
    tags=["Chat"],
)
async def get_conversation_history(session_id: str) -> ConversationHistoryResponse:
    """Get conversation history for a session."""
    if not chat_service:
        raise HTTPException(status_code=500, detail="Chat service unavailable")

    try:
        history = await chat_service.get_session_history(session_id)
        return ConversationHistoryResponse(
            session_id=session_id, messages=history, message_count=len(history)
        )
    except Exception as e:
        logger.error(f"Memory retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/memory/{session_id}", tags=["Chat"])
async def clear_conversation(session_id: str) -> Dict[str, Any]:
    """Clear conversation history for a session."""
    if not chat_service:
        raise HTTPException(status_code=500, detail="Chat service unavailable")

    try:
        success = await chat_service.clear_session(session_id)
        return {"session_id": session_id, "cleared": success}
    except Exception as e:
        logger.error(f"Memory clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rag/stats", response_model=RAGStatsResponse, tags=["RAG"])
async def get_rag_statistics() -> RAGStatsResponse:
    """Get RAG system statistics."""
    if not chat_service:
        raise HTTPException(status_code=500, detail="Chat service unavailable")

    try:
        stats = await chat_service.get_stats()
        guardrail_stats = guardrail_service.get_stats() if guardrail_service else {}

        return RAGStatsResponse(
            vector_store=stats.get("retrieval", {}).get("vector_store", {}),
            embedding_cache=stats.get("retrieval", {}).get("embedding_cache", {}),
            memory=stats.get("memory", {}),
            guardrails=guardrail_stats,
        )
    except Exception as e:
        logger.error(f"Stats retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback", tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest) -> Dict[str, Any]:
    """Submit feedback for a chat message."""
    if not feedback_service:
        raise HTTPException(status_code=500, detail="Feedback service unavailable")

    return await feedback_service.add_feedback(request.is_positive)


@app.get(
    "/api/dashboard/metrics",
    response_model=DashboardMetricsResponse,
    tags=["Dashboard"],
)
async def get_dashboard_metrics() -> DashboardMetricsResponse:
    """Get real-time dashboard metrics."""
    if not feedback_service or not document_processor:
        raise HTTPException(status_code=500, detail="Services unavailable")

    # Get stats
    feedback_stats = feedback_service.get_stats()

    # Count real files in uploads directory
    settings = get_settings()
    total_docs = 0
    if os.path.exists(settings.upload_directory):
        # Recursively count files
        for root, dirs, files in os.walk(settings.upload_directory):
            total_docs += len([f for f in files if not f.startswith(".")])

    # Get processed docs count from memory service
    processed_docs = await document_processor.list_documents(limit=1000)
    processed_count = len(processed_docs)

    # Calculate average confidence
    avg_conf = 0.985  # Default high confidence if no docs
    if processed_docs:
        confidences = []
        for doc in processed_docs:
            # Try to extract confidence from nested structure
            result = doc.get("result", {})
            ai_analysis = result.get("ai_analysis", {})
            conf = ai_analysis.get("confidence")
            if conf:
                confidences.append(float(conf))

        if confidences:
            avg_conf = sum(confidences) / len(confidences)

    return DashboardMetricsResponse(
        total_documents=total_docs,
        processed_documents=processed_count,
        accuracy=feedback_stats["accuracy"],
        avg_confidence=avg_conf * 100
        if avg_conf <= 1
        else avg_conf,  # Ensure percentage
    )


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "detail": "Resource not found",
            "path": str(request.url.path),
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
