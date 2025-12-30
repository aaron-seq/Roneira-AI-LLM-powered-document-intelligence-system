"""
Retrieval Service for RAG Pipeline

Provides high-level retrieval operations combining embeddings,
vector search, and result ranking for RAG applications.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio

from backend.services.embedding_service import EmbeddingService
from backend.services.vector_store_service import (
    VectorStoreService,
    VectorDocument,
    SearchResult,
)
from backend.services.text_splitter_service import TextSplitterService, TextChunk

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "chunks_indexed": self.chunks_indexed,
            "success": self.success,
            "error": self.error,
        }


class RetrievalService:
    """
    High-level retrieval service for RAG applications.

    Combines text splitting, embedding, and vector search
    into a unified retrieval pipeline.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize component services
        self.text_splitter = TextSplitterService(
            default_chunk_size=chunk_size, default_chunk_overlap=chunk_overlap
        )

        self.embedding_service = EmbeddingService(
            model_name=embedding_model, use_cache=True
        )

        self.vector_store = VectorStoreService(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_dimension=self.embedding_service.dimension,
        )

        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize all component services."""
        await self.embedding_service.initialize()
        await self.vector_store.initialize()
        self.is_initialized = True
        logger.info("RetrievalService initialized successfully")

    async def index_document(
        self,
        document_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        splitting_strategy: str = "recursive",
    ) -> IndexingResult:
        """
        Index a document for retrieval.

        Splits the document into chunks, generates embeddings,
        and stores them in the vector database.

        Args:
            document_id: Unique document identifier
            content: Document text content
            metadata: Optional document metadata
            splitting_strategy: Text splitting strategy

        Returns:
            IndexingResult with indexing status
        """
        if not self.is_initialized:
            await self.initialize()

        try:
            # Split document into chunks
            chunks = self.text_splitter.split_document(
                text=content, strategy=splitting_strategy, metadata=metadata or {}
            )

            if not chunks:
                return IndexingResult(
                    document_id=document_id, chunks_indexed=0, success=True
                )

            # Generate embeddings for all chunks
            chunk_texts = [c.content for c in chunks]
            embeddings = await self.embedding_service.embed_texts(chunk_texts)

            # Create vector documents
            vector_docs = []
            for i, (chunk, embedding_result) in enumerate(zip(chunks, embeddings)):
                chunk_metadata = {
                    **chunk.metadata,
                    "chunk_index": i,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                }

                vector_doc = VectorDocument(
                    chunk_id=f"{document_id}_chunk_{i}",
                    document_id=document_id,
                    content=chunk.content,
                    embedding=embedding_result.embedding,
                    metadata=chunk_metadata,
                )
                vector_docs.append(vector_doc)

            # Store in vector database
            await self.vector_store.add_documents(vector_docs)

            logger.info(f"Indexed document {document_id}: {len(vector_docs)} chunks")

            return IndexingResult(
                document_id=document_id, chunks_indexed=len(vector_docs), success=True
            )

        except Exception as e:
            logger.error(f"Failed to index document {document_id}: {e}")
            return IndexingResult(
                document_id=document_id, chunks_indexed=0, success=False, error=str(e)
            )

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        filter_dict: Optional[Dict[str, Any]] = None,
        max_context_length: int = 4000,
    ) -> RetrievalContext:
        """
        Retrieve relevant context for a query.

        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            min_score: Minimum similarity score threshold
            filter_dict: Optional metadata filter
            max_context_length: Maximum combined context length

        Returns:
            RetrievalContext with search results
        """
        if not self.is_initialized:
            await self.initialize()

        # Generate query embedding
        query_embedding = await self.embedding_service.embed_text(query)

        # Search vector store
        results = await self.vector_store.search(
            query_embedding=query_embedding.embedding,
            top_k=top_k,
            filter_dict=filter_dict,
        )

        # Filter by minimum score
        filtered_results = [r for r in results if r.score >= min_score]

        # Combine context with length limit
        combined_context = self._combine_context(filtered_results, max_context_length)

        # Calculate average score
        avg_score = (
            sum(r.score for r in filtered_results) / len(filtered_results)
            if filtered_results
            else 0.0
        )

        return RetrievalContext(
            query=query,
            results=filtered_results,
            combined_context=combined_context,
            total_chunks=len(filtered_results),
            avg_score=avg_score,
        )

    def _combine_context(self, results: List[SearchResult], max_length: int) -> str:
        """Combine search results into context string."""
        if not results:
            return ""

        context_parts = []
        current_length = 0

        for result in results:
            chunk_text = result.content

            if current_length + len(chunk_text) > max_length:
                # Truncate if necessary
                remaining = max_length - current_length
                if remaining > 100:  # Only add if meaningful
                    context_parts.append(chunk_text[:remaining] + "...")
                break

            context_parts.append(chunk_text)
            current_length += len(chunk_text) + 2  # +2 for newlines

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
        return await self.retrieve(
            query=query, top_k=top_k, filter_dict={"document_id": document_id}
        )

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
            "vector_store": vector_stats,
            "embedding_cache": cache_stats,
        }

    async def cleanup(self) -> None:
        """Clean up all resources."""
        await self.embedding_service.cleanup()
        await self.vector_store.cleanup()
        self.is_initialized = False
        logger.info("RetrievalService cleaned up")
