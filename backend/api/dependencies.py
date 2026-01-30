from functools import lru_cache
from typing import Optional

from backend.services.document_processor import DocumentProcessorService
from backend.services.chat_service import ChatService
from backend.services.auth_service import AuthService
from backend.services.feedback_service import FeedbackService
from backend.services.guardrail_service import GuardrailService
from backend.services.websocket_manager import WebSocketManager
from backend.core.database import DatabaseManager

# Global Instances (Singleton Pattern)
_document_processor: Optional[DocumentProcessorService] = None
_chat_service: Optional[ChatService] = None
_auth_service: Optional[AuthService] = None
_feedback_service: Optional[FeedbackService] = None
_guardrail_service: Optional[GuardrailService] = None
_websocket_manager: Optional[WebSocketManager] = None
_db_manager: Optional[DatabaseManager] = None

async def initialize_services():
    """Initialize all global services. Called on startup."""
    global _document_processor, _chat_service, _auth_service, _feedback_service
    global _guardrail_service, _websocket_manager, _db_manager

    _db_manager = DatabaseManager()
    await _db_manager.initialize()

    _websocket_manager = WebSocketManager()
    
    _document_processor = DocumentProcessorService()
    await _document_processor.initialize()

    _auth_service = AuthService()

    _chat_service = ChatService()
    await _chat_service.initialize()
    
    _guardrail_service = GuardrailService()
    _feedback_service = FeedbackService()

async def cleanup_services():
    """Cleanup services on shutdown."""
    if _db_manager:
        await _db_manager.close()

def get_document_processor() -> DocumentProcessorService:
    if not _document_processor:
        raise RuntimeError("Document Processor not initialized")
    return _document_processor

def get_chat_service() -> ChatService:
    if not _chat_service:
        raise RuntimeError("Chat Service not initialized")
    return _chat_service

def get_auth_service() -> AuthService:
    if not _auth_service:
        raise RuntimeError("Auth Service not initialized")
    return _auth_service

def get_feedback_service() -> FeedbackService:
    if not _feedback_service:
        # Feedback service might fail init, handle gracefully or raise
        if not _feedback_service:
             pass # Logic to handle optional service or raise
    return _feedback_service

def get_guardrail_service() -> GuardrailService:
    return _guardrail_service

def get_websocket_manager() -> WebSocketManager:
    if not _websocket_manager:
        raise RuntimeError("WebSocket Manager not initialized")
    return _websocket_manager

def get_db_manager() -> DatabaseManager:
    if not _db_manager:
        raise RuntimeError("Database Manager not initialized")
    return _db_manager
