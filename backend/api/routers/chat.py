from fastapi import APIRouter, HTTPException, Depends, status
from typing import Dict, Any, List
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
from backend.services.chat_service import ChatService
from backend.services.guardrail_service import GuardrailService
from backend.api.dependencies import get_chat_service, get_guardrail_service
import logging

logger = logging.getLogger(__name__)
# Prefix is handled in __init__.py but some routes are /api/search which doesn't fit /chat prefix perfectly if we use prefix="/chat".
# In __init__.py I did: api_router.include_router(chat_router, tags=["Chat"]) (No prefix)
# So I should define full paths here or group them.

router = APIRouter()

@router.post("/chat", response_model=ChatAPIResponse)
async def chat_completion(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
    guardrail_service: GuardrailService = Depends(get_guardrail_service)
) -> ChatAPIResponse:
    """RAG-powered chat completion endpoint."""
    try:
        # Validate input with guardrails
        # Guardrail service might be optional or always present. 
        # In dependencies it returns Optional[GuardrailService] ?? No, just instance.
        
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


@router.post("/search", response_model=SearchResponse, tags=["RAG"])
async def semantic_search(
    request: SearchRequest,
    chat_service: ChatService = Depends(get_chat_service)
) -> SearchResponse:
    """Semantic search over indexed documents."""
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


@router.post("/documents/{document_id}/index", response_model=IndexDocumentResponse, tags=["RAG"])
async def index_document_for_rag(
    document_id: str, 
    request: IndexDocumentRequest,
    chat_service: ChatService = Depends(get_chat_service)
) -> IndexDocumentResponse:
    """Index a document for RAG retrieval."""
    try:
        result = await chat_service.index_document(
            document_id=document_id, content=request.content, metadata=request.metadata
        )

        return IndexDocumentResponse(**result)
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/{session_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    session_id: str,
    chat_service: ChatService = Depends(get_chat_service)
) -> ConversationHistoryResponse:
    """Get conversation history for a session."""
    try:
        history = await chat_service.get_session_history(session_id)
        return ConversationHistoryResponse(
            session_id=session_id, messages=history, message_count=len(history)
        )
    except Exception as e:
        logger.error(f"Memory retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/memory/{session_id}")
async def clear_conversation(
    session_id: str,
    chat_service: ChatService = Depends(get_chat_service)
) -> Dict[str, Any]:
    """Clear conversation history for a session."""
    try:
        success = await chat_service.clear_session(session_id)
        return {"session_id": session_id, "cleared": success}
    except Exception as e:
        logger.error(f"Memory clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag/stats", response_model=RAGStatsResponse, tags=["RAG"])
async def get_rag_statistics(
    chat_service: ChatService = Depends(get_chat_service),
    guardrail_service: GuardrailService = Depends(get_guardrail_service)
) -> RAGStatsResponse:
    """Get RAG system statistics."""
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
