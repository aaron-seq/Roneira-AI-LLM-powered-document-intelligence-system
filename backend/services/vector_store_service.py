"""
Vector Store Service for Semantic Search

Provides vector storage and kNN similarity search using ChromaDB
with fallback to in-memory storage for development.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import ChromaDB, provide fallback
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None
    logger.warning(
        "ChromaDB not installed. Using in-memory vector store. "
        "Install with: pip install chromadb"
    )

import numpy as np
from numpy.linalg import norm


@dataclass
class SearchResult:
    """Result from a similarity search."""

    document_id: str
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class VectorDocument:
    """Document to be stored in vector database."""

    chunk_id: str
    document_id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]


class InMemoryVectorStore:
    """
    Simple in-memory vector store for development.
    Uses cosine similarity for search.
    """

    def __init__(self):
        self.documents: Dict[str, VectorDocument] = {}
        self.embeddings: Dict[str, List[float]] = {}

    def add_documents(self, documents: List[VectorDocument]) -> None:
        """Add documents to the store."""
        for doc in documents:
            self.documents[doc.chunk_id] = doc
            self.embeddings[doc.chunk_id] = doc.embedding

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar documents."""
        if not self.documents:
            return []

        query_vec = np.array(query_embedding)
        scores = []

        for chunk_id, embedding in self.embeddings.items():
            doc = self.documents[chunk_id]

            # Apply filter if provided
            if filter_dict:
                match = all(doc.metadata.get(k) == v for k, v in filter_dict.items())
                if not match:
                    continue

            # Cosine similarity
            doc_vec = np.array(embedding)
            similarity = np.dot(query_vec, doc_vec) / (
                norm(query_vec) * norm(doc_vec) + 1e-10
            )
            scores.append((chunk_id, float(similarity)))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for chunk_id, score in scores[:top_k]:
            doc = self.documents[chunk_id]
            results.append(
                SearchResult(
                    document_id=doc.document_id,
                    chunk_id=doc.chunk_id,
                    content=doc.content,
                    score=score,
                    metadata=doc.metadata,
                )
            )

        return results

    def delete_by_document_id(self, document_id: str) -> int:
        """Delete all chunks for a document."""
        to_delete = [
            chunk_id
            for chunk_id, doc in self.documents.items()
            if doc.document_id == document_id
        ]

        for chunk_id in to_delete:
            del self.documents[chunk_id]
            del self.embeddings[chunk_id]

        return len(to_delete)

    def get_document_count(self) -> int:
        """Get total number of stored chunks."""
        return len(self.documents)

    def clear(self) -> None:
        """Clear all documents."""
        self.documents.clear()
        self.embeddings.clear()


class VectorStoreService:
    """
    Main service for vector storage and similarity search.

    Uses ChromaDB when available, falls back to in-memory store.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None,
        embedding_dimension: int = 384,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_dimension = embedding_dimension
        self.is_initialized = False

        self._client = None
        self._collection = None
        self._memory_store: Optional[InMemoryVectorStore] = None
        self._use_chromadb = CHROMADB_AVAILABLE

    async def initialize(self) -> None:
        """Initialize the vector store."""
        if self._use_chromadb:
            await self._initialize_chromadb()
        else:
            self._memory_store = InMemoryVectorStore()
            logger.info("VectorStoreService initialized with in-memory store")

        self.is_initialized = True

    async def _initialize_chromadb(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            loop = asyncio.get_event_loop()

            if self.persist_directory:
                Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
                self._client = await loop.run_in_executor(
                    None, lambda: chromadb.PersistentClient(path=self.persist_directory)
                )
            else:
                self._client = await loop.run_in_executor(None, chromadb.Client)

            # Get or create collection
            self._collection = await loop.run_in_executor(
                None,
                lambda: self._client.get_or_create_collection(
                    name=self.collection_name, metadata={"hnsw:space": "cosine"}
                ),
            )

            logger.info(
                f"VectorStoreService initialized with ChromaDB "
                f"(collection: {self.collection_name})"
            )

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            logger.info("Falling back to in-memory store")
            self._use_chromadb = False
            self._memory_store = InMemoryVectorStore()

    async def add_documents(self, documents: List[VectorDocument]) -> int:
        """
        Add documents to the vector store.

        Args:
            documents: List of VectorDocument objects

        Returns:
            Number of documents added
        """
        if not self.is_initialized:
            await self.initialize()

        if not documents:
            return 0

        if self._use_chromadb and self._collection:
            return await self._add_to_chromadb(documents)
        else:
            self._memory_store.add_documents(documents)
            return len(documents)

    async def _add_to_chromadb(self, documents: List[VectorDocument]) -> int:
        """Add documents to ChromaDB."""
        loop = asyncio.get_event_loop()

        ids = [doc.chunk_id for doc in documents]
        embeddings = [doc.embedding for doc in documents]
        documents_text = [doc.content for doc in documents]
        metadatas = [
            {**doc.metadata, "document_id": doc.document_id} for doc in documents
        ]

        await loop.run_in_executor(
            None,
            lambda: self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents_text,
                metadatas=metadatas,
            ),
        )

        return len(documents)

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filter

        Returns:
            List of SearchResult objects
        """
        if not self.is_initialized:
            await self.initialize()

        if self._use_chromadb and self._collection:
            return await self._search_chromadb(query_embedding, top_k, filter_dict)
        else:
            return self._memory_store.search(query_embedding, top_k, filter_dict)

    async def _search_chromadb(
        self,
        query_embedding: List[float],
        top_k: int,
        filter_dict: Optional[Dict[str, Any]],
    ) -> List[SearchResult]:
        """Search in ChromaDB."""
        loop = asyncio.get_event_loop()

        where_clause = filter_dict if filter_dict else None

        results = await loop.run_in_executor(
            None,
            lambda: self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            ),
        )

        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                # ChromaDB returns distances, convert to similarity
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - distance  # Cosine distance to similarity

                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                document_id = metadata.pop("document_id", "unknown")

                search_results.append(
                    SearchResult(
                        document_id=document_id,
                        chunk_id=chunk_id,
                        content=results["documents"][0][i]
                        if results["documents"]
                        else "",
                        score=score,
                        metadata=metadata,
                    )
                )

        return search_results

    async def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks for a document.

        Args:
            document_id: ID of the document to delete

        Returns:
            Number of chunks deleted
        """
        if not self.is_initialized:
            return 0

        if self._use_chromadb and self._collection:
            loop = asyncio.get_event_loop()

            # Get existing chunks for this document
            existing = await loop.run_in_executor(
                None, lambda: self._collection.get(where={"document_id": document_id})
            )

            if existing["ids"]:
                await loop.run_in_executor(
                    None, lambda: self._collection.delete(ids=existing["ids"])
                )
                return len(existing["ids"])
            return 0
        else:
            return self._memory_store.delete_by_document_id(document_id)

    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        if not self.is_initialized:
            return {"initialized": False}

        if self._use_chromadb and self._collection:
            loop = asyncio.get_event_loop()
            count = await loop.run_in_executor(None, lambda: self._collection.count())
            return {
                "initialized": True,
                "backend": "chromadb",
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory,
            }
        else:
            return {
                "initialized": True,
                "backend": "in_memory",
                "document_count": self._memory_store.get_document_count(),
            }

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._memory_store:
            self._memory_store.clear()

        self._collection = None
        self._client = None
        self.is_initialized = False
        logger.info("VectorStoreService cleaned up")
