"""RAG chat, semantic search, indexing and conversation memory endpoints."""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status

from backend.api.dependencies import (
    get_chat_service,
    get_document_repository,
    get_guardrail_service,
)
from backend.api.security import CurrentUser, get_current_user
from backend.models.chat_models import (
    ChatMessageModel,
    ChatRequest,
    ChatSource,
    ConversationHistoryResponse,
    IndexDocumentRequest,
    IndexDocumentResponse,
    RAGStatsResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from backend.models.chat_models import (
    ChatResponse as ChatAPIResponse,
)
from backend.repositories.document_repository import DocumentRepository
from backend.services.chat_service import ChatService
from backend.services.guardrail_service import GuardrailService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatAPIResponse,
    tags=["Chat"],
    summary="Ask a grounded question",
)
async def chat_completion(
    request: ChatRequest,
    user: CurrentUser = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
    guardrail_service: GuardrailService = Depends(get_guardrail_service),
    repository: DocumentRepository = Depends(get_document_repository),
) -> ChatAPIResponse:
    """Answer a question using only documents the caller owns.

    Retrieval is restricted to the caller's own documents; without that a user
    could ask questions whose answers are drawn from another tenant's files.
    """
    if guardrail_service:
        validation = await guardrail_service.validate_input(request.message)
        if not validation.is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Input validation failed: {validation.details}",
            )
        message = validation.filtered_content or request.message
    else:
        message = request.message

    allowed_ids = await repository.owned_document_ids(user.user_id)

    try:
        response = await chat_service.chat(
            message=message,
            session_id=request.session_id,
            use_rag=request.use_rag,
            rag_top_k=request.rag_top_k,
            document_filter=request.document_id,
            owner_id=user.user_id,
            allowed_document_ids=allowed_ids,
        )
    except Exception as exc:
        logger.exception("Chat request failed for user %s", user.user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to generate a response. Please try again.",
        ) from exc

    return ChatAPIResponse(
        message=response.message,
        session_id=response.session_id,
        sources=[ChatSource.model_validate(source) for source in response.sources],
        model=response.model,
        grounded=bool(response.usage.get("grounded")),
        embeddings_are_real=bool(response.usage.get("embeddings_are_real")),
    )


@router.post(
    "/search",
    response_model=SearchResponse,
    tags=["RAG"],
    summary="Semantic search over your documents",
)
async def semantic_search(
    request: SearchRequest,
    user: CurrentUser = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
    repository: DocumentRepository = Depends(get_document_repository),
) -> SearchResponse:
    """Search indexed chunks and return them with their citation metadata.

    ``document_id`` and ``min_score`` from the request are applied here; they
    were previously accepted by the schema and then ignored, so a
    document-scoped search silently searched everything.
    """
    allowed_ids = await repository.owned_document_ids(user.user_id)

    try:
        results = await chat_service.search_documents(
            query=request.query,
            top_k=request.top_k,
            min_score=request.min_score,
            owner_id=user.user_id,
            allowed_document_ids=allowed_ids,
            document_id=request.document_id,
        )
    except Exception as exc:
        logger.exception("Search failed for user %s", user.user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed. Please try again.",
        ) from exc

    return SearchResponse(
        query=request.query,
        results=[SearchResult.model_validate(r) for r in results],
        total_results=len(results),
        embeddings_are_real=chat_service.retrieval.embeddings_are_real,
    )


@router.post(
    "/documents/{document_id}/index",
    response_model=IndexDocumentResponse,
    tags=["RAG"],
    summary="Re-index a document you own",
)
async def index_document_for_rag(
    document_id: str,
    request: IndexDocumentRequest,
    user: CurrentUser = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
    repository: DocumentRepository = Depends(get_document_repository),
) -> IndexDocumentResponse:
    """Index supplied content against an existing document you own."""
    document = await repository.get(document_id, owner_id=user.user_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )

    result = await chat_service.index_document(
        document_id=document_id,
        content=request.content,
        metadata={**(request.metadata or {}), "filename": document.filename},
        owner_id=user.user_id,
    )

    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("error") or "Indexing failed",
        )

    return IndexDocumentResponse(**result)


@router.get(
    "/memory/{session_id}",
    response_model=ConversationHistoryResponse,
    tags=["Chat"],
    summary="Read conversation history",
)
async def get_conversation_history(
    session_id: str,
    user: CurrentUser = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
) -> ConversationHistoryResponse:
    """Return the message history for a session."""
    history = await chat_service.get_session_history(session_id)
    return ConversationHistoryResponse(
        session_id=session_id,
        messages=[ChatMessageModel.model_validate(m) for m in history],
        message_count=len(history),
    )


@router.delete(
    "/memory/{session_id}", tags=["Chat"], summary="Clear conversation history"
)
async def clear_conversation(
    session_id: str,
    user: CurrentUser = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
) -> Dict[str, Any]:
    """Forget a conversation."""
    cleared = await chat_service.clear_session(session_id)
    return {"session_id": session_id, "cleared": cleared}


@router.get(
    "/rag/stats",
    response_model=RAGStatsResponse,
    tags=["RAG"],
    summary="Retrieval subsystem statistics",
)
async def get_rag_statistics(
    user: CurrentUser = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
    guardrail_service: GuardrailService = Depends(get_guardrail_service),
) -> RAGStatsResponse:
    """Report index size, embedding backend, and guardrail activity."""
    stats = await chat_service.get_stats()
    retrieval = stats.get("retrieval", {})

    return RAGStatsResponse(
        vector_store=retrieval.get("vector_store", {}),
        embedding_cache=retrieval.get("embedding_cache", {}),
        memory=stats.get("memory", {}),
        guardrails=guardrail_service.get_stats() if guardrail_service else {},
        embeddings_are_real=bool(retrieval.get("embeddings_are_real")),
        degraded_reason=retrieval.get("degraded_reason"),
    )
