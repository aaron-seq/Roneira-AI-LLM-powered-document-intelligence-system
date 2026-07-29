"""
Chat and RAG API Models

Pydantic models for chat completion, RAG retrieval,
and document indexing API endpoints.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

# ============ Chat Models ============


class ChatMessageModel(BaseModel):
    """A single chat message."""

    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"role": "user", "content": "What is this document about?"}
        }
    )


class ChatRequest(BaseModel):
    """Request for chat completion."""

    message: str = Field(..., description="User message", min_length=1)
    session_id: Optional[str] = Field(
        None, description="Session ID for conversation continuity"
    )
    use_rag: bool = Field(True, description="Whether to use RAG for context")
    rag_top_k: int = Field(5, description="Number of chunks to retrieve", ge=1, le=20)
    document_id: Optional[str] = Field(None, description="Filter to specific document")
    stream: bool = Field(False, description="Stream response")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Summarize the main points of this document",
                "session_id": "session-123",
                "use_rag": True,
                "rag_top_k": 5,
            }
        }
    )


class ChatSource(BaseModel):
    """A retrieved chunk cited in an answer.

    ``page_number`` and ``filename`` are what let a user verify a claim
    against the original document rather than trusting an opaque chunk id.
    """

    document_id: str
    chunk_id: str
    score: float
    content_preview: str
    page_number: Optional[int] = Field(
        default=None, description="Page the chunk came from, when paginated"
    )
    filename: Optional[str] = Field(default=None, description="Source filename")


class ChatResponse(BaseModel):
    """Response from chat completion."""

    message: str = Field(..., description="Assistant response")
    session_id: str = Field(..., description="Session ID")
    sources: List[ChatSource] = Field(
        default_factory=list, description="RAG sources used"
    )
    model: str = Field(..., description="Model used for generation")
    grounded: bool = Field(
        default=False,
        description=(
            "True when the answer was built from retrieved document context. "
            "False means the model answered without supporting sources."
        ),
    )
    embeddings_are_real: bool = Field(
        default=False,
        description=(
            "False when retrieval ran on placeholder vectors because no "
            "embedding model was available; treat sources as unreliable."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Based on the document, the main points are...",
                "session_id": "session-123",
                "sources": [
                    {
                        "document_id": "doc-1",
                        "chunk_id": "doc-1_chunk_0",
                        "score": 0.85,
                        "content_preview": "The document discusses...",
                    }
                ],
                "model": "llama3.2:3b",
            }
        }
    )


# ============ RAG Models ============


class SearchRequest(BaseModel):
    """Request for semantic search."""

    query: str = Field(..., description="Search query", min_length=1)
    top_k: int = Field(5, description="Number of results", ge=1, le=50)
    document_id: Optional[str] = Field(None, description="Filter to specific document")
    min_score: float = Field(0.0, description="Minimum similarity score", ge=0, le=1)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"query": "financial projections for 2024", "top_k": 10}
        }
    )


class SearchResult(BaseModel):
    """Single search result."""

    document_id: str
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Response from semantic search."""

    query: str
    results: List[SearchResult]
    total_results: int
    embeddings_are_real: bool = Field(
        default=False,
        description="False when results come from placeholder vectors.",
    )


class IndexDocumentRequest(BaseModel):
    """Request to index a document for RAG."""

    document_id: Optional[str] = Field(
        default=None,
        description="Ignored; the document is identified by the URL path.",
    )
    content: str = Field(..., description="Document content to index", min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")
    chunk_size: int = Field(1000, description="Chunk size for splitting", ge=100, le=5000)
    chunk_overlap: int = Field(200, description="Overlap between chunks", ge=0, le=1000)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "doc-123",
                "content": "Full document text content...",
                "metadata": {"filename": "report.pdf", "author": "John Doe"},
            }
        }
    )


class IndexDocumentResponse(BaseModel):
    """Response from document indexing."""

    document_id: str
    chunks_indexed: int
    success: bool
    error: Optional[str] = None
    embeddings_are_real: bool = False


# ============ Memory Models ============


class ConversationHistoryResponse(BaseModel):
    """Response containing conversation history."""

    session_id: str
    messages: List[ChatMessageModel]
    message_count: int


class SessionSummaryResponse(BaseModel):
    """Summary of a conversation session."""

    session_id: str
    exists: bool
    message_count: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# ============ Statistics Models ============


class RAGStatsResponse(BaseModel):
    """Statistics for RAG system."""

    vector_store: Dict[str, Any]
    embedding_cache: Dict[str, Any]
    memory: Dict[str, Any]
    guardrails: Dict[str, Any]
    embeddings_are_real: bool = False
    degraded_reason: Optional[str] = Field(
        default=None,
        description="Why real embeddings are unavailable, when applicable.",
    )
