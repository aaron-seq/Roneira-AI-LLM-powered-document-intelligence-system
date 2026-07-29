"""Document processing pipeline.

Flow for a single upload:

    validate -> stream to disk -> persist record -> extract text
             -> LLM enrichment -> chunk + embed + index -> persist result

Two behaviours changed relative to the original implementation and both are
load-bearing for the product:

* Processing state is written to the database, not a per-process dictionary,
  so a restart or a second worker does not lose (or hide) uploads.
* The source file is retained by default. Deleting it made every citation
  unverifiable — a user could be shown "page 7 of contract.pdf" with no way
  to open page 7. Retention is configurable for deployments that must not
  keep customer documents at rest.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.common.helpers import extract_text_from_file
from backend.core.config import get_settings
from backend.core.models import DocumentStatus
from backend.observability.structured_logging import get_logger, with_correlation_id
from backend.repositories.document_repository import DocumentRepository
from backend.services.file_validation import (
    ValidatedUpload,
    save_upload_stream,
)
from backend.services.local_llm_service import LocalLLMService
from backend.services.retrieval_service import RetrievalService

logger = get_logger(__name__)


class DocumentProcessorService:
    """Owns the document lifecycle from upload to indexed and queryable."""

    def __init__(
        self,
        repository: DocumentRepository,
        retrieval_service: RetrievalService,
        llm_service: Optional[LocalLLMService] = None,
    ):
        self.settings = get_settings()
        self.repository = repository
        # Shared with ChatService: one index, one embedding model, one cache.
        self.retrieval_service = retrieval_service
        self.llm_service = llm_service or LocalLLMService()

    # ------------------------------------------------------------- lifecycle
    @with_correlation_id
    async def initialize(self) -> None:
        """Prepare the LLM and retrieval dependencies.

        LLM failure is tolerated (documents still get extracted and indexed);
        retrieval failure is not, because without it the product does nothing.
        """
        try:
            await self.llm_service.initialize()
        except Exception as exc:
            logger.warning(
                "LLM unavailable; documents will be extracted and indexed "
                "without AI enrichment",
                error=str(exc),
            )

        await self.retrieval_service.initialize()

        if self.llm_service.is_initialized:
            logger.info("Document processor ready (AI enrichment enabled)")
        else:
            logger.warning("Document processor ready in BASIC mode (no AI enrichment)")

    @property
    def ai_enabled(self) -> bool:
        """True when LLM enrichment is available."""
        return self.llm_service.is_initialized

    # ----------------------------------------------------------------- intake
    async def accept_upload(
        self,
        file,
        owner_id: str,
        processing_options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, ValidatedUpload]:
        """Validate and store an upload, returning its new document id.

        Args:
            file: Starlette ``UploadFile``.
            owner_id: Authenticated owner of the document.
            processing_options: Options recorded against the document.

        Returns:
            ``(document_id, ValidatedUpload)``.

        Raises:
            FileValidationError: if the upload is rejected.
        """
        document_id = str(uuid.uuid4())
        destination = self._storage_path(owner_id, document_id, file.filename)

        stored = await save_upload_stream(
            upload=file,
            destination=destination,
            max_bytes=self.settings.max_file_size,
        )

        await self.repository.create(
            document_id=document_id,
            owner_id=owner_id,
            filename=os.path.basename(file.filename),
            size_bytes=stored.size_bytes,
            checksum=stored.checksum,
            content_type=stored.detected_type,
            file_extension=stored.extension,
            storage_path=stored.path,
            processing_options=processing_options or {},
        )

        logger.info(
            "Accepted upload",
            document_id=document_id,
            owner_id=owner_id,
            size_bytes=stored.size_bytes,
            detected_type=stored.detected_type,
        )
        return document_id, stored

    def _storage_path(self, owner_id: str, document_id: str, filename: str) -> str:
        """Build a collision-free storage path under the owner's directory.

        The stored name is derived from the document id, never from the
        user-supplied filename, so the original name cannot influence the path.
        """
        extension = os.path.splitext(filename or "")[1].lower()
        safe_owner = "".join(c for c in owner_id if c.isalnum() or c in "-_") or "unknown"
        directory = os.path.join(self.settings.upload_directory, safe_owner)
        return os.path.join(directory, f"{document_id}{extension}")

    # ------------------------------------------------------------ processing
    @with_correlation_id
    async def process_document(
        self,
        document_id: str,
        owner_id: str,
        file_path: str,
        options: Dict[str, Any],
        filename: str,
        progress_callback: Optional[Callable] = None,
    ) -> None:
        """Run the full pipeline for one document.

        Never raises: failures are recorded on the document row and pushed to
        the progress callback, because this runs detached in a background task
        where an exception would otherwise vanish into the void.
        """
        started = time.perf_counter()
        file_extension = os.path.splitext(filename)[1].lstrip(".").lower() or "unknown"

        try:
            await self._advance(
                document_id,
                DocumentStatus.PROCESSING,
                10,
                "Extracting text",
                progress_callback,
            )

            # Raises TextExtractionError with a message written for the user;
            # the handler below records it on the document.
            extracted_text, doc_metadata = await extract_text_from_file(file_path)
            doc_metadata = {**doc_metadata, "filename": filename}

            await self._advance(
                document_id,
                DocumentStatus.PROCESSING,
                40,
                "Text extracted, analysing with AI",
                progress_callback,
            )

            ai_analysis = await self._enrich(extracted_text, doc_metadata, options)

            await self._advance(
                document_id,
                DocumentStatus.PROCESSING,
                75,
                "Indexing for search",
                progress_callback,
            )

            indexing = await self.retrieval_service.index_document(
                document_id=document_id,
                content=extracted_text,
                metadata={
                    "filename": filename,
                    "document_id": document_id,
                    **{
                        # ChromaDB metadata values must be scalars.
                        k: v
                        for k, v in doc_metadata.items()
                        if isinstance(v, (str, int, float, bool))
                    },
                },
                owner_id=owner_id,
            )

            if not indexing.success:
                raise RuntimeError(f"Indexing failed: {indexing.error}")

            await self.repository.replace_chunks(
                document_id, owner_id, indexing.chunk_records
            )

            await self.repository.mark_completed(
                document_id,
                extracted_text=extracted_text,
                doc_metadata=doc_metadata,
                ai_analysis=ai_analysis,
                chunk_count=indexing.chunks_indexed,
                embeddings_are_real=indexing.embeddings_are_real,
            )

            await self._maybe_delete_source(document_id, file_path)
            await self._notify(document_id, progress_callback)

            duration = time.perf_counter() - started
            self._record_metrics("success", file_extension, duration)
            logger.info(
                "Document processed",
                document_id=document_id,
                chunks=indexing.chunks_indexed,
                duration_seconds=round(duration, 2),
            )

        except Exception as exc:
            logger.error(
                "Document processing failed",
                document_id=document_id,
                error=str(exc),
                exc_info=True,
            )
            await self.repository.mark_failed(document_id, str(exc))
            await self._maybe_delete_source(document_id, file_path)
            await self._notify(document_id, progress_callback)
            self._record_metrics("error", file_extension, time.perf_counter() - started)

    async def _enrich(
        self,
        extracted_text: str,
        doc_metadata: Dict[str, Any],
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run LLM enrichment, degrading to a text-only result on failure."""
        if not options.get("enhance_with_ai", True):
            return {
                "summary": "AI enrichment was disabled for this document.",
                "key_points": [],
                "entities": [],
                "enrichment": "skipped",
            }

        if not self.ai_enabled:
            return {
                "summary": (
                    "Text extracted and indexed. AI enrichment was unavailable "
                    "(no LLM configured), so no summary or entities were produced."
                ),
                "key_points": [],
                "entities": [],
                "enrichment": "unavailable",
            }

        result = await self.llm_service.enhance_document_data(
            extracted_text, doc_metadata
        )
        result.setdefault("enrichment", "completed")
        # The full text is already stored on the document row; keeping a second
        # copy inside the JSON column doubled the storage for no benefit.
        result.pop("enhanced_text", None)
        return result

    # --------------------------------------------------------------- helpers
    async def _advance(
        self,
        document_id: str,
        status: DocumentStatus,
        progress: int,
        message: str,
        callback: Optional[Callable],
    ) -> None:
        """Persist a progress step and push it to subscribers."""
        await self.repository.update_progress(
            document_id, status=status, progress=progress, message=message
        )
        await self._notify(document_id, callback)

    async def _notify(self, document_id: str, callback: Optional[Callable]) -> None:
        """Send the current document state to a progress callback."""
        if callback is None:
            return
        document = await self.repository.get(document_id)
        if document is None:
            return
        try:
            result = callback(document.to_status_dict())
            if result is not None and hasattr(result, "__await__"):
                await result
        except Exception as exc:
            # A disconnected websocket must not fail the pipeline.
            logger.debug(f"Progress callback failed: {exc}")

    async def _maybe_delete_source(self, document_id: str, file_path: str) -> None:
        """Delete the source file when retention is disabled."""
        if self.settings.retain_source_files:
            return
        try:
            os.remove(file_path)
            await self.repository.clear_storage_path(document_id)
            logger.info(
                "Source file removed (retention disabled)", document_id=document_id
            )
        except FileNotFoundError:
            await self.repository.clear_storage_path(document_id)
        except OSError as exc:
            logger.warning(f"Could not remove source file: {exc}")

    @staticmethod
    def _record_metrics(status: str, document_type: str, duration: float) -> None:
        """Publish processing outcome, never failing the pipeline."""
        try:
            from backend.observability.metrics import get_metrics

            get_metrics().record_document_processed(
                status=status, document_type=document_type, latency_seconds=duration
            )
        except Exception as exc:  # pragma: no cover - metrics are best-effort
            logger.debug(f"Could not record processing metrics: {exc}")

    # ------------------------------------------------------------- retention
    async def purge_expired_sources(self) -> int:
        """Delete retained source files past ``SOURCE_RETENTION_DAYS``.

        Returns:
            Number of files removed.
        """
        expired = await self.repository.expired_source_files(
            self.settings.source_retention_days
        )
        removed = 0
        for document_id, path in expired:
            try:
                os.remove(path)
                removed += 1
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning(f"Retention purge failed for {path}: {exc}")
                continue
            await self.repository.clear_storage_path(document_id)

        if removed:
            logger.info(f"Retention purge removed {removed} source files")
        return removed

    # ------------------------------------------------------------ read paths
    async def get_document_status(
        self, document_id: str, owner_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Return the current status of a document, or None if not visible."""
        document = await self.repository.get(document_id, owner_id=owner_id)
        return document.to_status_dict() if document else None

    async def get_document(
        self, document_id: str, owner_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Return the full document record including extracted text."""
        document = await self.repository.get(document_id, owner_id=owner_id)
        return document.to_status_dict(include_text=True) if document else None

    async def list_documents(
        self,
        limit: int = 20,
        offset: int = 0,
        status_filter: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Return a page of documents plus the total matching count."""
        documents, total = await self.repository.list(
            owner_id=owner_id,
            limit=limit,
            offset=offset,
            status_filter=status_filter,
        )
        return [d.to_status_dict() for d in documents], total

    async def delete_document(self, document_id: str, owner_id: str) -> bool:
        """Delete a document, its chunks, its vectors and its source file."""
        document = await self.repository.get(document_id, owner_id=owner_id)
        if document is None:
            return False

        storage_path = document.storage_path
        await self.retrieval_service.delete_document(document_id)
        deleted = await self.repository.delete(document_id, owner_id=owner_id)

        if deleted and storage_path:
            try:
                os.remove(storage_path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning(f"Could not delete source file {storage_path}: {exc}")

        return deleted

    async def stats(self, owner_id: Optional[str] = None) -> Dict[str, Any]:
        """Aggregate document counts for dashboards."""
        return await self.repository.stats(owner_id=owner_id)
