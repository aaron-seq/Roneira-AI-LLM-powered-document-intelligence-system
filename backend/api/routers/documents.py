from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks, status, WebSocket, WebSocketDisconnect
from typing import Optional
from backend.Models.responses import DocumentUploadResponse, DocumentStatusResponse
from backend.services.document_processor import DocumentProcessorService
from backend.services.websocket_manager import WebSocketManager
from backend.api.dependencies import get_document_processor, get_websocket_manager
from backend.core.config import get_settings
import uuid
import os
import asyncio
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document_for_processing(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file to process"),
    extract_tables: bool = Form(default=True),
    extract_images: bool = Form(default=False),
    enhance_with_ai: bool = Form(default=True),
    document_processor: DocumentProcessorService = Depends(get_document_processor),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
) -> DocumentUploadResponse:
    """Enhanced document upload with comprehensive validation"""

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No file provided"
        )

    allowed_extensions = {".pdf", ".doc", ".docx", ".txt", ".png", ".jpg", ".jpeg"}
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file_extension}. Allowed: {', '.join(allowed_extensions)}",
        )

    settings = get_settings()
    content = await file.read()
    await file.seek(0)
    
    if len(content) > settings.max_file_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.max_file_size // (1024 * 1024)}MB",
        )

    document_id = str(uuid.uuid4())

    try:
        # Save file securely
        file_path = await document_processor.save_uploaded_file(
            file,
            document_id,
            "demo_user",  # In production, get from JWT token
        )

        # Initialize status synchronously
        document_processor.initialize_status(document_id, file.filename)

        # Queue for background processing
        processing_options = {
            "extract_tables": extract_tables,
            "extract_images": extract_images,
            "enhance_with_ai": enhance_with_ai,
            "user_id": "demo_user",
        }

        # Need to define the background task function wrapper or import it
        # The background task needs access to services.
        # Since BackgroundTasks runs after response, we need to pass the service instance or ensure it's available.
        # Ideally, we pass the method of the bound instance.
        
        background_tasks.add_task(
            _process_document_task,
            document_processor,
            websocket_manager,
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

@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_processing_status(
    document_id: str,
    document_processor: DocumentProcessorService = Depends(get_document_processor)
) -> DocumentStatusResponse:
    """Get real-time document processing status"""
    try:
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

@router.get("/")
async def list_documents(
    limit: int = 10, 
    offset: int = 0, 
    status_filter: Optional[str] = None,
    document_processor: DocumentProcessorService = Depends(get_document_processor)
):
    """List processed documents with pagination"""
    try:
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

# NOTE: WebSockets usually go on the main app or a router, but here using router is fine.
# The endpoint path is relative to the router prefix. Main.py mounted it at /documents
# So ws url will be /api/documents/ws/documents/{client_id} -> A bit repetitive.
# In main.py it was /ws/documents/{client_id}.
# I should probably put this websocket route in a separate place or keep the path clean.
# Let's assign path "/ws/{client_id}" inside this router. 
# Caller: /api/documents/ws/... might be confusing. 
# BUT, main.py has `api_router` with prefix `/documents`.
# I'll stick to specific WS route or put it in general if needed.
# Let's put it here for now: `/ws/{client_id}`.

@router.websocket("/ws/{client_id}")
async def websocket_document_updates(
    websocket: WebSocket, 
    client_id: str,
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """Enhanced WebSocket for real-time updates"""
    try:
        await websocket_manager.connect(websocket, client_id)
        while True:
            message = await websocket.receive_text()
            await websocket_manager.handle_client_message(client_id, message)
    except WebSocketDisconnect:
        await websocket_manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        await websocket_manager.disconnect(client_id)


# Internal Task Wrapper
async def _process_document_task(
    processor: DocumentProcessorService,
    ws_manager: WebSocketManager,
    document_id: str, 
    file_path: str, 
    options: dict, 
    filename: str
):
    """Background task wrapper"""
    try:
        await processor.process_document_with_ai(
            document_id,
            file_path,
            options,
            filename=filename,
            progress_callback=lambda progress: asyncio.create_task(
                ws_manager.broadcast_progress(document_id, progress)
            ),
        )
    except Exception as e:
        logger.error(f"Document processing failed for {document_id}: {e}")
        await ws_manager.broadcast_error(document_id, str(e))
