"""Persistence for documents and their retrievable chunks."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence

from sqlalchemy import delete, func, select

from backend.core.database import DatabaseManager
from backend.core.models import Document, DocumentChunk, DocumentStatus

logger = logging.getLogger(__name__)


class DocumentRepository:
    """CRUD operations for the document lifecycle.

    Every read that can be attributed to a user takes an ``owner_id`` and
    filters on it. Ownership is enforced here rather than in the routers so a
    new endpoint cannot accidentally leak another tenant's documents.
    """

    def __init__(self, db_manager: DatabaseManager):
        self._db = db_manager

    # ---------------------------------------------------------------- create
    async def create(
        self,
        *,
        document_id: str,
        owner_id: str,
        filename: str,
        size_bytes: int,
        checksum: str,
        content_type: Optional[str] = None,
        file_extension: Optional[str] = None,
        storage_path: Optional[str] = None,
        processing_options: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """Record a newly accepted upload in the ``queued`` state."""
        async with self._db.get_session() as session:
            document = Document(
                id=document_id,
                owner_id=owner_id,
                filename=filename,
                size_bytes=size_bytes,
                checksum=checksum,
                content_type=content_type,
                file_extension=file_extension,
                storage_path=storage_path,
                status=DocumentStatus.QUEUED,
                progress=0,
                message="Queued for processing",
                processing_options=processing_options or {},
                doc_metadata={},
                ai_analysis={},
            )
            session.add(document)
            await session.flush()
            await session.refresh(document)
            return document

    # ------------------------------------------------------------------ read
    async def get(
        self, document_id: str, owner_id: Optional[str] = None
    ) -> Optional[Document]:
        """Fetch one document, scoped to ``owner_id`` when supplied."""
        async with self._db.get_session() as session:
            stmt = select(Document).where(Document.id == document_id)
            if owner_id is not None:
                stmt = stmt.where(Document.owner_id == owner_id)
            return (await session.execute(stmt)).scalar_one_or_none()

    async def list(
        self,
        *,
        owner_id: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        status_filter: Optional[str] = None,
    ) -> tuple[List[Document], int]:
        """Return a page of documents plus the total matching count.

        The total is what lets a client render real pagination; the previous
        implementation reported ``total = len(page)``, so the UI could never
        tell whether more documents existed.
        """
        async with self._db.get_session() as session:
            stmt = select(Document)
            count_stmt = select(func.count()).select_from(Document)

            if owner_id is not None:
                stmt = stmt.where(Document.owner_id == owner_id)
                count_stmt = count_stmt.where(Document.owner_id == owner_id)

            if status_filter:
                try:
                    status = DocumentStatus(status_filter)
                except ValueError as exc:
                    raise ValueError(
                        f"Unknown status '{status_filter}'. Valid values: "
                        + ", ".join(s.value for s in DocumentStatus)
                    ) from exc
                stmt = stmt.where(Document.status == status)
                count_stmt = count_stmt.where(Document.status == status)

            stmt = stmt.order_by(Document.created_at.desc()).limit(limit).offset(offset)

            documents = list((await session.execute(stmt)).scalars().all())
            total = (await session.execute(count_stmt)).scalar_one()
            return documents, total

    async def get_chunks(
        self, document_id: str, chunk_ids: Optional[Sequence[str]] = None
    ) -> List[DocumentChunk]:
        """Load chunk citation metadata, optionally restricted to ``chunk_ids``."""
        async with self._db.get_session() as session:
            stmt = select(DocumentChunk).where(DocumentChunk.document_id == document_id)
            if chunk_ids is not None:
                stmt = stmt.where(DocumentChunk.id.in_(list(chunk_ids)))
            stmt = stmt.order_by(DocumentChunk.chunk_index)
            return list((await session.execute(stmt)).scalars().all())

    async def resolve_chunks(self, chunk_ids: Sequence[str]) -> Dict[str, DocumentChunk]:
        """Look up chunks by id across documents, keyed by chunk id.

        Used to enrich chat citations with page numbers and offsets.
        """
        if not chunk_ids:
            return {}
        async with self._db.get_session() as session:
            stmt = select(DocumentChunk).where(DocumentChunk.id.in_(list(chunk_ids)))
            rows = (await session.execute(stmt)).scalars().all()
            return {row.id: row for row in rows}

    async def owned_document_ids(self, owner_id: str) -> List[str]:
        """All document ids belonging to a user (for retrieval scoping)."""
        async with self._db.get_session() as session:
            stmt = select(Document.id).where(Document.owner_id == owner_id)
            return list((await session.execute(stmt)).scalars().all())

    # ---------------------------------------------------------------- update
    async def update_progress(
        self,
        document_id: str,
        *,
        status: Optional[DocumentStatus] = None,
        progress: Optional[int] = None,
        message: Optional[str] = None,
    ) -> Optional[Document]:
        """Advance the processing state machine."""
        async with self._db.get_session() as session:
            document = await session.get(Document, document_id)
            if document is None:
                return None
            if status is not None:
                document.status = status
            if progress is not None:
                document.progress = max(0, min(100, progress))
            if message is not None:
                document.message = message
            await session.flush()
            await session.refresh(document)
            return document

    async def mark_completed(
        self,
        document_id: str,
        *,
        extracted_text: str,
        doc_metadata: Dict[str, Any],
        ai_analysis: Dict[str, Any],
        chunk_count: int,
        embeddings_are_real: bool,
    ) -> Optional[Document]:
        """Store the processing result and flip the document to ``completed``."""
        async with self._db.get_session() as session:
            document = await session.get(Document, document_id)
            if document is None:
                return None
            document.status = DocumentStatus.COMPLETED
            document.progress = 100
            document.message = "Processing complete"
            document.error = None
            document.extracted_text = extracted_text
            document.doc_metadata = doc_metadata
            document.ai_analysis = ai_analysis
            document.chunk_count = chunk_count
            document.embeddings_are_real = embeddings_are_real
            document.page_count = doc_metadata.get("pages")
            document.word_count = doc_metadata.get("word_count")
            document.completed_at = datetime.utcnow()
            await session.flush()
            await session.refresh(document)
            return document

    async def mark_failed(self, document_id: str, error: str) -> Optional[Document]:
        """Record a processing failure with its reason."""
        async with self._db.get_session() as session:
            document = await session.get(Document, document_id)
            if document is None:
                return None
            document.status = DocumentStatus.FAILED
            document.message = "Processing failed"
            # Errors can carry stack-trace-sized payloads; keep them bounded.
            document.error = error[:4000]
            document.completed_at = datetime.utcnow()
            await session.flush()
            await session.refresh(document)
            return document

    async def clear_storage_path(self, document_id: str) -> None:
        """Forget where a source file lived (after deleting it)."""
        async with self._db.get_session() as session:
            document = await session.get(Document, document_id)
            if document is not None:
                document.storage_path = None

    async def replace_chunks(
        self, document_id: str, owner_id: str, chunks: Iterable[Dict[str, Any]]
    ) -> int:
        """Replace a document's chunk rows; re-indexing must not duplicate."""
        async with self._db.get_session() as session:
            await session.execute(
                delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
            )
            count = 0
            for chunk in chunks:
                session.add(
                    DocumentChunk(
                        id=chunk["chunk_id"],
                        document_id=document_id,
                        owner_id=owner_id,
                        chunk_index=chunk["chunk_index"],
                        content=chunk["content"],
                        page_number=chunk.get("page_number"),
                        start_char=chunk.get("start_char", 0),
                        end_char=chunk.get("end_char", 0),
                        token_estimate=chunk.get("token_estimate", 0),
                    )
                )
                count += 1
            return count

    # ---------------------------------------------------------------- delete
    async def delete(self, document_id: str, owner_id: Optional[str] = None) -> bool:
        """Delete a document and its chunks. Returns False if not found."""
        async with self._db.get_session() as session:
            stmt = select(Document).where(Document.id == document_id)
            if owner_id is not None:
                stmt = stmt.where(Document.owner_id == owner_id)
            document = (await session.execute(stmt)).scalar_one_or_none()
            if document is None:
                return False
            await session.delete(document)
            return True

    async def expired_source_files(self, older_than_days: int) -> List[tuple[str, str]]:
        """Return ``(document_id, storage_path)`` for files past retention."""
        if older_than_days <= 0:
            return []
        cutoff = datetime.utcnow() - timedelta(days=older_than_days)
        async with self._db.get_session() as session:
            stmt = select(Document.id, Document.storage_path).where(
                Document.storage_path.is_not(None), Document.created_at < cutoff
            )
            return [(row[0], row[1]) for row in (await session.execute(stmt)).all()]

    # ----------------------------------------------------------- aggregates
    async def stats(self, owner_id: Optional[str] = None) -> Dict[str, Any]:
        """Counts by status plus indexing totals, for the dashboard."""
        async with self._db.get_session() as session:
            stmt = select(Document.status, func.count()).group_by(Document.status)
            totals_stmt = select(
                func.coalesce(func.sum(Document.chunk_count), 0),
                func.coalesce(func.sum(Document.word_count), 0),
            )
            if owner_id is not None:
                stmt = stmt.where(Document.owner_id == owner_id)
                totals_stmt = totals_stmt.where(Document.owner_id == owner_id)

            by_status = {
                status.value if isinstance(status, DocumentStatus) else str(status): count
                for status, count in (await session.execute(stmt)).all()
            }
            total_chunks, total_words = (await session.execute(totals_stmt)).one()

            return {
                "by_status": by_status,
                "total_documents": sum(by_status.values()),
                "total_chunks": int(total_chunks or 0),
                "total_words": int(total_words or 0),
            }
