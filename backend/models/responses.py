"""Pydantic response models for the Document Intelligence API.

These models are the API contract: they define what the OpenAPI schema
advertises and what clients can rely on.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------- system


class ComponentHealth(BaseModel):
    """Health of a single dependency."""

    status: str = Field(description="ok, degraded, or unavailable")
    detail: Optional[str] = Field(
        default=None, description="Why the component is not fully healthy"
    )


class HealthCheckResponse(BaseModel):
    """Response model for the health check endpoint.

    ``status`` is ``degraded`` when the service is answering requests but a
    capability is missing (no LLM, placeholder embeddings). Reporting
    "healthy" in that state hides exactly the failures worth paging on.
    """

    status: str = Field(description="healthy, degraded, or unhealthy")
    version: str = Field(description="Application version")
    environment: str = Field(default="development")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    components: Dict[str, ComponentHealth] = Field(default_factory=dict)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "degraded",
                "version": "2.1.0",
                "environment": "development",
                "timestamp": "2025-01-15T10:30:00Z",
                "components": {
                    "database": {"status": "ok"},
                    "embeddings": {
                        "status": "degraded",
                        "detail": "keyword-only: sentence-transformers is not installed",
                    },
                    "llm": {"status": "unavailable", "detail": "Ollama unreachable"},
                },
            }
        }
    )


# -------------------------------------------------------------- documents


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""

    document_id: str = Field(description="Unique document identifier")
    status: str = Field(default="queued")
    message: str = Field(default="Document accepted and queued for processing")
    filename: str = Field(description="Original filename")
    size_bytes: int = Field(default=0, description="Accepted size in bytes")
    checksum: Optional[str] = Field(
        default=None, description="SHA-256 of the uploaded bytes"
    )
    detected_type: Optional[str] = Field(
        default=None, description="Content type detected from the file's bytes"
    )
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "queued",
                "message": "Document accepted and queued for processing",
                "filename": "invoice.pdf",
                "size_bytes": 84213,
                "checksum": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
                "detected_type": "application/pdf",
                "upload_timestamp": "2025-01-15T10:30:00Z",
            }
        }
    )


class DocumentStatusResponse(BaseModel):
    """Processing state of a document."""

    document_id: str
    status: str = Field(description="queued, processing, completed, or failed")
    progress: int = Field(default=0, ge=0, le=100)
    message: Optional[str] = None
    filename: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = Field(
        default=None, description="Failure reason when status is 'failed'"
    )

    page_count: Optional[int] = None
    word_count: Optional[int] = None
    chunk_count: int = Field(default=0, description="Indexed retrievable chunks")
    size_bytes: int = 0
    checksum: Optional[str] = None

    embeddings_are_real: bool = Field(
        default=False,
        description=(
            "False when the document was indexed with the keyword-only "
            "lexical fallback because no embedding model was available; "
            "paraphrased questions will not match it."
        ),
    )
    ai_insights: Optional[Dict[str, Any]] = Field(
        default=None, description="Summary, key points, entities and action items"
    )

    model_config = ConfigDict(extra="ignore")


class DocumentDetailResponse(DocumentStatusResponse):
    """A document plus its extracted text and full analysis."""

    result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Extracted text, document metadata and AI analysis",
    )


class DocumentListResponse(BaseModel):
    """A page of documents.

    ``total`` is the count of all matching documents, not the size of this
    page, so clients can render real pagination.
    """

    documents: List[DocumentStatusResponse]
    total: int
    limit: int
    offset: int
    has_more: bool = False


class DocumentDeleteResponse(BaseModel):
    """Confirmation that a document was removed."""

    document_id: str
    deleted: bool


class DocumentChange(BaseModel):
    """One difference between two documents.

    Both sides are quoted verbatim, with the page each came from, so a change
    can be checked against the originals the same way a citation can.
    """

    kind: str = Field(description="added, removed or changed")
    left: List[str] = Field(
        default_factory=list, description="Paragraphs in the first document"
    )
    right: List[str] = Field(
        default_factory=list, description="Paragraphs in the second document"
    )
    left_page: Optional[int] = None
    right_page: Optional[int] = None


class DocumentComparisonResponse(BaseModel):
    """Paragraph-level differences between two documents.

    Derived entirely from the extracted text of both documents; no model is
    involved, so this result does not degrade when the LLM is unavailable.
    """

    left_document_id: str
    right_document_id: str
    left_filename: Optional[str] = None
    right_filename: Optional[str] = None
    changes: List[DocumentChange] = Field(default_factory=list)
    added: int = 0
    removed: int = 0
    changed: int = 0
    unchanged_paragraphs: int = 0
    left_paragraphs: int = 0
    right_paragraphs: int = 0
    similarity: float = Field(
        default=1.0,
        description="1.0 when the documents are paragraph-for-paragraph identical",
    )
    truncated: bool = Field(
        default=False, description="True when more changes exist than were returned"
    )


# ------------------------------------------------------------------- auth


class AuthTokenResponse(BaseModel):
    """Response model for the authentication endpoint."""

    access_token: str = Field(description="JWT access token")
    token_type: str = Field(default="bearer")
    expires_in: int = Field(default=1800, description="Token lifetime in seconds")
    user_id: Optional[str] = None
    username: Optional[str] = None
    roles: List[str] = Field(default_factory=list)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIs...",
                "token_type": "bearer",
                "expires_in": 1800,
                "user_id": "demo_user_001",
                "username": "demo",
                "roles": ["user"],
            }
        }
    )


class CurrentUserResponse(BaseModel):
    """The identity attached to the presented token."""

    user_id: str
    username: str
    roles: List[str] = Field(default_factory=list)
    is_anonymous: bool = False


# ------------------------------------------------------------------ error


class ErrorResponse(BaseModel):
    """Standard error envelope.

    ``correlation_id`` matches the ``X-Correlation-ID`` response header, so a
    user-reported error can be found in the logs directly.
    """

    error: str = Field(description="Error class name")
    message: str = Field(description="Human-readable error message")
    details: Dict[str, Any] = Field(default_factory=dict)
    correlation_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "ResourceNotFoundError",
                "message": "Document with id abc123 not found",
                "details": {},
                "correlation_id": "0f5c2a1e-4e2b-4a9c-9d3a-1f2e3d4c5b6a",
                "timestamp": "2025-01-15T10:30:00Z",
            }
        }
    )
