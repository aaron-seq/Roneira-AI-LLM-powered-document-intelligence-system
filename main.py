"""
Main FastAPI Application for the LLM Document Intelligence System

This file defines the core FastAPI application, including its routes, middleware,
and lifecycle events. It serves as the main entry point for the backend API.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
import json
import os

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
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import application_settings as settings
from database import get_db_session, initialize_database
from llm_service import llm_service
from azure_document_intelligence import azure_document_intelligence_service
from redis_client import redis_client
from auth import create_access_token, get_current_user

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- WebSocket Connection Manager ---
class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)
        
        # Clean up disconnected connections
        for conn in disconnected:
            self.disconnect(conn)


connection_manager = ConnectionManager()


# --- Pydantic Models for API Requests and Responses ---
class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime


class Token(BaseModel):
    access_token: str
    token_type: str


# --- Application Lifecycle Events ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events."""
    logger.info("Starting up the LLM Document Intelligence System...")
    # Ensure upload directory exists on startup
    os.makedirs(settings.upload_directory, exist_ok=True)

    await initialize_database()
    await redis_client.connect()
    await llm_service.initialize()
    await azure_document_intelligence_service.initialize()

    yield

    logger.info("Shutting down...")
    await redis_client.disconnect()


# --- FastAPI App Initialization ---
app = FastAPI(
    title="LLM-Powered Document Intelligence System",
    version="1.0.0",
    lifespan=lifespan,
)

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Provides a health check endpoint for monitoring."""
    return HealthResponse(status="ok", version="1.0.0", timestamp=datetime.utcnow())


@app.post("/auth/login", response_model=Token)
async def login(username: str = Form(...), password: str = Form(...)):
    """Authenticates a user and returns a JWT access token."""
    # In a real application, you would verify the username and password here.
    access_token = create_access_token(data={"sub": username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/api/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user),
):
    """Uploads a document and queues it for asynchronous processing."""
    try:
        document_id = f"doc_{datetime.utcnow().timestamp()}"
        file_path = os.path.join(
            settings.upload_directory, f"{document_id}_{file.filename}"
        )

        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        background_tasks.add_task(
            process_document_in_background, document_id, file_path, current_user
        )

        return {"document_id": document_id, "status": "queued"}

    except Exception as e:
        logger.error(f"Error uploading document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to upload document.")


@app.get("/api/documents/{document_id}/status")
async def get_document_status(document_id: str):
    """Retrieves the processing status of a document."""
    status = await redis_client.get(f"doc_status:{document_id}")
    if not status:
        raise HTTPException(status_code=404, detail="Document not found.")
    return status


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """Provides a WebSocket endpoint for real-time updates."""
    await connection_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)


# --- Background Processing Task ---
async def process_document_in_background(document_id: str, file_path: str, user: str):
    """
    The background task that processes the uploaded document.
    It updates the status in Redis and broadcasts updates via WebSockets.
    """
    status_key = f"doc_status:{document_id}"

    try:
        # 1. Update status to 'processing'
        await redis_client.set(status_key, {"status": "processing", "progress": 10})
        await connection_manager.broadcast(
            json.dumps({"document_id": document_id, "status": "processing"})
        )

        # 2. Analyze with Azure Document Intelligence
        # FIXED: Changed azure__result to azure_result (removed double underscore)
        azure_result = (
            await azure_document_intelligence_service.analyze_document_from_file(
                file_path
            )
        )
        await redis_client.set(status_key, {"status": "processing", "progress": 50})

        # 3. Enhance with LLM - now uses correctly named variable
        llm_result = await llm_service.enhance_extracted_data(azure_result)
        await redis_client.set(status_key, {"status": "processing", "progress": 90})

        # 4. Store final result and update status to 'completed'
        final_result = {
            "status": "completed",
            "data": llm_result.get("data"),
            "processed_at": datetime.utcnow().isoformat(),
        }
        await redis_client.set(f"doc_result:{document_id}", final_result)
        await redis_client.set(status_key, {"status": "completed", "progress": 100})
        await connection_manager.broadcast(
            json.dumps(
                {
                    "document_id": document_id,
                    "status": "completed",
                    "data": final_result,
                }
            )
        )

    except Exception as e:
        logger.error(f"Error processing document {document_id}: {e}", exc_info=True)
        error_status = {"status": "failed", "error": str(e)}
        await redis_client.set(status_key, error_status)
        await connection_manager.broadcast(
            json.dumps({"document_id": document_id, "status": "failed", "error": str(e)})
        )
        raise  # Re-raise to ensure proper error logging

    finally:
        # Clean up the temporary file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError as e:
                logger.warning(f"Failed to remove temporary file {file_path}: {e}")
