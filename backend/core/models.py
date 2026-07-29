"""SQLAlchemy models for durable application state.

Before this module existed the document lifecycle lived in a per-process
dictionary: a restart (or a second worker) lost every upload, and the
``/documents`` listing showed whatever happened to be in *that* worker's
memory. Anything a user can see in the UI is persisted here instead.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.core.database import Base


class DocumentStatus(str, enum.Enum):
    """Lifecycle of an uploaded document."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(Base):
    """An uploaded document and the result of processing it."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    content_type: Mapped[Optional[str]] = mapped_column(String(128))
    file_extension: Mapped[Optional[str]] = mapped_column(String(16))
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    #: SHA-256 of the uploaded bytes. Lets us detect re-uploads of the same
    #: document and gives every answer a verifiable provenance anchor.
    checksum: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    #: Where the source file is retained, or NULL when retention is disabled.
    storage_path: Mapped[Optional[str]] = mapped_column(String(1024))

    status: Mapped[DocumentStatus] = mapped_column(
        SAEnum(DocumentStatus, native_enum=False, length=16),
        default=DocumentStatus.QUEUED,
        nullable=False,
        index=True,
    )
    progress: Mapped[int] = mapped_column(Integer, default=0)
    message: Mapped[Optional[str]] = mapped_column(String(1024))
    error: Mapped[Optional[str]] = mapped_column(Text)

    page_count: Mapped[Optional[int]] = mapped_column(Integer)
    word_count: Mapped[Optional[int]] = mapped_column(Integer)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)

    extracted_text: Mapped[Optional[str]] = mapped_column(Text)
    doc_metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    ai_analysis: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    processing_options: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    #: True when the chunks were embedded with a real model, false when the
    #: deterministic development fallback was used. Surfaced in the API so a
    #: user is never shown pseudo-embedding results as if they were real.
    embeddings_are_real: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    chunks: Mapped[List["DocumentChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        Index("ix_documents_owner_status", "owner_id", "status"),
        Index("ix_documents_owner_created", "owner_id", "created_at"),
    )

    @property
    def is_terminal(self) -> bool:
        return self.status in (DocumentStatus.COMPLETED, DocumentStatus.FAILED)

    def to_status_dict(self, include_text: bool = False) -> Dict[str, Any]:
        """Shape used by the status/list API responses."""
        payload: Dict[str, Any] = {
            "document_id": self.id,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "filename": self.filename,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "page_count": self.page_count,
            "word_count": self.word_count,
            "chunk_count": self.chunk_count,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "embeddings_are_real": self.embeddings_are_real,
            "ai_insights": self.ai_analysis or None,
        }
        if self.error:
            payload["error"] = self.error
        if include_text:
            payload["result"] = {
                "original_text": self.extracted_text or "",
                "metadata": self.doc_metadata or {},
                "ai_analysis": self.ai_analysis or {},
                "processing_options": self.processing_options or {},
            }
        return payload


class DocumentChunk(Base):
    """A retrievable slice of a document.

    The embedding vector lives in the vector store; this table keeps the
    citation metadata (page, character offsets, ordinal) so an answer can
    point back at an exact location in the source document even if the
    vector store is rebuilt.
    """

    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    owner_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    page_number: Mapped[Optional[int]] = mapped_column(Integer)
    start_char: Mapped[int] = mapped_column(Integer, default=0)
    end_char: Mapped[int] = mapped_column(Integer, default=0)
    token_estimate: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (Index("ix_chunks_doc_index", "document_id", "chunk_index"),)


class ChatFeedback(Base):
    """Thumbs up/down on an assistant answer.

    Feedback previously lived in a counter on a singleton service, so the
    "accuracy" number on the dashboard reset on every deploy and could not be
    attributed to a message, a user, or a point in time.
    """

    __tablename__ = "chat_feedback"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    message_id: Mapped[str] = mapped_column(String(128), index=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(128), index=True)
    owner_id: Mapped[Optional[str]] = mapped_column(String(128), index=True)
    is_positive: Mapped[bool] = mapped_column(Boolean, nullable=False)
    comment: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, index=True
    )


__all__ = [
    "ChatFeedback",
    "Document",
    "DocumentChunk",
    "DocumentStatus",
]
