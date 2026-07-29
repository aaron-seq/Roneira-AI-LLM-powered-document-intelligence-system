"""
Retrieval Service for RAG Pipeline

Provides high-level retrieval operations combining embeddings,
vector search, and result ranking for RAG applications.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.core.config import get_settings
from backend.services.embedding_service import EmbeddingService
from backend.services.text_splitter_service import TextSplitterService
from backend.services.vector_store_service import (
    SearchResult,
    VectorDocument,
    VectorStoreService,
)

logger = logging.getLogger(__name__)


@dataclass
class RetrievalContext:
    """Context retrieved for a query."""

    query: str
    results: List[SearchResult]
    combined_context: str
    total_chunks: int
    avg_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "combined_context": self.combined_context,
            "total_chunks": self.total_chunks,
            "avg_score": self.avg_score,
        }


@dataclass
class IndexingResult:
    """Result of document indexing operation."""

    document_id: str
    chunks_indexed: int
    success: bool
    error: Optional[str] = None
    embeddings_are_real: bool = False
    #: Citation records for the indexed chunks, in document order. Persisted
    #: alongside the document so an answer can cite an exact page and offset
    #: even after the vector store is rebuilt.
    chunk_records: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "chunks_indexed": self.chunks_indexed,
            "success": self.success,
            "error": self.error,
            "embeddings_are_real": self.embeddings_are_real,
        }


class RetrievalService:
    """
    High-level retrieval service for RAG applications.

    Combines text splitting, embedding, and vector search
    into a unified retrieval pipeline.
    """

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        require_real_embeddings: Optional[bool] = None,
    ):
        # Defaults come from configuration rather than hard-coded literals, so
        # EMBEDDING_MODEL / CHUNK_SIZE / VECTOR_STORE_PATH — all documented in
        # .env.example — actually take effect.
        settings = get_settings()

        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        )
        self.min_score = settings.retrieval_min_score

        self.text_splitter = TextSplitterService(
            default_chunk_size=self.chunk_size,
            default_chunk_overlap=self.chunk_overlap,
        )

        self.embedding_service = EmbeddingService(
            model_name=embedding_model or settings.embedding_model,
            use_cache=True,
            require_real_model=(
                settings.require_real_embeddings
                if require_real_embeddings is None
                else require_real_embeddings
            ),
        )

        # A persist directory is the default. Without one ChromaDB is ephemeral,
        # so every restart silently dropped the entire index.
        self.persist_directory = (
            persist_directory
            if persist_directory is not None
            else settings.vector_store_path
        )

        self.vector_store = VectorStoreService(
            collection_name=collection_name or settings.vector_collection_name,
            persist_directory=self.persist_directory,
            embedding_dimension=self.embedding_service.dimension,
        )

        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize all component services."""
        if self.is_initialized:
            return
        await self.embedding_service.initialize()
        await self.vector_store.initialize()
        self.is_initialized = True

        if not self.embedding_service.is_real:
            logger.warning(
                "RetrievalService initialized with LEXICAL matching only: %s",
                self.embedding_service.degraded_reason,
            )
        else:
            logger.info(
                "RetrievalService initialized (model=%s, store=%s)",
                self.embedding_service.model_name,
                self.persist_directory or "ephemeral",
            )

    @property
    def embeddings_are_real(self) -> bool:
        """True when the index was built with a genuine embedding model."""
        return self.embedding_service.is_real

    async def index_document(
        self,
        document_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        splitting_strategy: str = "recursive",
        owner_id: Optional[str] = None,
    ) -> IndexingResult:
        """
        Index a document for retrieval.

        Splits the document into chunks, generates embeddings, and stores them
        in the vector database. Re-indexing the same ``document_id`` replaces
        the previous chunks rather than adding duplicates.

        Args:
            document_id: Unique document identifier
            content: Document text content
            metadata: Optional document metadata
            splitting_strategy: Text splitting strategy
            owner_id: Owner stamped on each chunk so retrieval can be scoped
                to the requesting user.

        Returns:
            IndexingResult with indexing status and citation records.
        """
        if not self.is_initialized:
            await self.initialize()

        try:
            chunks = self.text_splitter.split_document(
                text=content, strategy=splitting_strategy, metadata=metadata or {}
            )

            if not chunks:
                return IndexingResult(
                    document_id=document_id,
                    chunks_indexed=0,
                    success=True,
                    embeddings_are_real=self.embeddings_are_real,
                )

            # Replace any previous index entries for this document so a
            # re-upload or re-index does not leave stale chunks behind.
            await self.vector_store.delete_document(document_id)

            chunk_texts = [c.content for c in chunks]
            embeddings = await self.embedding_service.embed_texts(chunk_texts)

            page_offsets = _page_offsets(content)

            vector_docs: List[VectorDocument] = []
            chunk_records: List[Dict[str, Any]] = []

            for i, (chunk, embedding_result) in enumerate(
                zip(chunks, embeddings, strict=True)
            ):
                chunk_id = f"{document_id}_chunk_{i}"
                page_number = _page_for_offset(page_offsets, chunk.start_char)

                chunk_metadata = {
                    **chunk.metadata,
                    "chunk_index": i,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                }
                if page_number is not None:
                    chunk_metadata["page_number"] = page_number
                if owner_id:
                    chunk_metadata["owner_id"] = owner_id

                vector_docs.append(
                    VectorDocument(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        # The page marker is captured in metadata above; keeping
                        # it in the body would put "--- Page 3 ---" into every
                        # citation preview and into the model's context.
                        content=_strip_page_markers(chunk.content),
                        embedding=embedding_result.embedding,
                        metadata=chunk_metadata,
                    )
                )
                chunk_records.append(
                    {
                        "chunk_id": chunk_id,
                        "chunk_index": i,
                        "content": _strip_page_markers(chunk.content),
                        "page_number": page_number,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        # Rough but stable estimate; ~4 characters per token.
                        "token_estimate": max(1, len(chunk.content) // 4),
                    }
                )

            await self.vector_store.add_documents(vector_docs)

            logger.info(
                "Indexed document %s: %s chunks (embeddings_real=%s)",
                document_id,
                len(vector_docs),
                self.embeddings_are_real,
            )

            return IndexingResult(
                document_id=document_id,
                chunks_indexed=len(vector_docs),
                success=True,
                embeddings_are_real=self.embeddings_are_real,
                chunk_records=chunk_records,
            )

        except Exception as exc:
            logger.exception("Failed to index document %s", document_id)
            return IndexingResult(
                document_id=document_id,
                chunks_indexed=0,
                success=False,
                error=str(exc),
                embeddings_are_real=self.embeddings_are_real,
            )

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        max_context_length: int = 4000,
        owner_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
    ) -> RetrievalContext:
        """
        Retrieve relevant context for a query.

        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            min_score: Minimum similarity score. Defaults to
                ``RETRIEVAL_MIN_SCORE``; pass 0.0 to disable thresholding.
            filter_dict: Optional metadata filter
            max_context_length: Maximum combined context length
            owner_id: Restrict results to chunks owned by this user
            document_ids: Restrict results to this set of documents

        Returns:
            RetrievalContext with search results
        """
        if not self.is_initialized:
            await self.initialize()

        threshold = self.min_score if min_score is None else min_score

        query_embedding = await self.embedding_service.embed_text(query)

        # Over-fetch so post-filtering by score/ownership can still return
        # top_k results instead of silently returning fewer.
        fetch_k = top_k if (owner_id is None and document_ids is None) else top_k * 4

        results = await self.vector_store.search(
            query_embedding=query_embedding.embedding,
            top_k=max(fetch_k, top_k),
            filter_dict=filter_dict,
        )

        allowed = set(document_ids) if document_ids is not None else None
        filtered_results: List[SearchResult] = []
        for result in results:
            if result.score < threshold:
                continue
            if allowed is not None and result.document_id not in allowed:
                continue
            if owner_id is not None:
                chunk_owner = result.metadata.get("owner_id")
                # Chunks indexed before ownership existed carry no owner; the
                # document_ids allow-list is the authoritative check for those.
                if chunk_owner is not None and chunk_owner != owner_id:
                    continue
            filtered_results.append(result)
            if len(filtered_results) >= top_k:
                break

        combined_context = self._combine_context(filtered_results, max_context_length)

        avg_score = (
            sum(r.score for r in filtered_results) / len(filtered_results)
            if filtered_results
            else 0.0
        )

        self._record_retrieval_metrics(filtered_results)

        return RetrievalContext(
            query=query,
            results=filtered_results,
            combined_context=combined_context,
            total_chunks=len(filtered_results),
            avg_score=avg_score,
        )

    @staticmethod
    def _record_retrieval_metrics(results: List[SearchResult]) -> None:
        """Publish retrieval outcome to Prometheus, never failing the query."""
        try:
            from backend.observability.metrics import get_metrics

            get_metrics().record_retrieval(
                results_found=len(results),
                top_score=results[0].score if results else None,
            )
        except Exception as exc:  # pragma: no cover - metrics are best-effort
            logger.debug("Could not record retrieval metrics: %s", exc)

    def _combine_context(self, results: List[SearchResult], max_length: int) -> str:
        """Combine search results into a labelled context block.

        Each chunk is prefixed with its citation marker so the model can
        reference a specific source, and the whole block is fenced with an
        explicit boundary. Untrusted document text was previously pasted
        straight into the prompt, where instructions inside a document read
        exactly like instructions from the operator.
        """
        if not results:
            return ""

        context_parts: List[str] = []
        current_length = 0

        for index, result in enumerate(results, start=1):
            page = result.metadata.get("page_number")
            locator = f", page {page}" if page else ""
            header = f"[Source {index}: {result.metadata.get('filename', result.document_id)}{locator}]"
            chunk_text = result.content

            budget = max_length - current_length - len(header) - 2
            if budget <= 100:
                break
            if len(chunk_text) > budget:
                chunk_text = chunk_text[:budget] + "..."

            context_parts.append(f"{header}\n{chunk_text}")
            current_length += len(header) + len(chunk_text) + 2

        return "\n\n".join(context_parts)

    async def retrieve_for_document(
        self, query: str, document_id: str, top_k: int = 5
    ) -> RetrievalContext:
        """
        Retrieve context from a specific document.

        Args:
            query: Search query
            document_id: Document to search within
            top_k: Number of chunks to retrieve

        Returns:
            RetrievalContext with results from specified document
        """
        return await self.retrieve(query=query, top_k=top_k, document_ids=[document_id])

    async def delete_document(self, document_id: str) -> bool:
        """
        Remove a document from the index.

        Args:
            document_id: Document to remove

        Returns:
            True if successful
        """
        if not self.is_initialized:
            return False

        try:
            deleted = await self.vector_store.delete_document(document_id)
            logger.info(f"Deleted document {document_id}: {deleted} chunks removed")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get retrieval service statistics."""
        vector_stats = await self.vector_store.get_stats()
        cache_stats = self.embedding_service.get_cache_stats()

        return {
            "initialized": self.is_initialized,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_score": self.min_score,
            "embeddings_are_real": self.embeddings_are_real,
            "degraded_reason": self.embedding_service.degraded_reason,
            "vector_store": vector_stats,
            "embedding_cache": cache_stats,
        }

    async def cleanup(self) -> None:
        """Clean up all resources."""
        await self.embedding_service.cleanup()
        await self.vector_store.cleanup()
        self.is_initialized = False
        logger.info("RetrievalService cleaned up")


# --------------------------------------------------------------------------
# Page tracking
#
# The PDF extractor emits "--- Page N ---" separators. Mapping chunk offsets
# back to those markers is what turns a retrieved chunk into a citation a user
# can verify ("page 7 of the contract") rather than an opaque chunk id.
# --------------------------------------------------------------------------

_PAGE_MARKER = re.compile(r"^--- Page (\d+) ---$", re.MULTILINE)


def _strip_page_markers(text: str) -> str:
    """Remove page separators from text shown to users or sent to the model."""
    return re.sub(r"\n{2,}", "\n\n", _PAGE_MARKER.sub("", text)).strip()


def _page_offsets(content: str) -> List[tuple]:
    """Return ``(character_offset, page_number)`` pairs, in document order."""
    return [
        (match.start(), int(match.group(1))) for match in _PAGE_MARKER.finditer(content)
    ]


def _page_for_offset(offsets: List[tuple], position: int) -> Optional[int]:
    """Return the page number containing ``position``, or None if unpaginated."""
    page: Optional[int] = None
    for offset, number in offsets:
        if offset <= position:
            page = number
        else:
            break
    return page
