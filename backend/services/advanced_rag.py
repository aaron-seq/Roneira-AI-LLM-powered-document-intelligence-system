"""
Advanced RAG Pipeline with Reranking.

Provides enhanced retrieval-augmented generation with:
- Multi-stage retrieval
- Cross-encoder reranking
- Query expansion
- Context compression
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk retrieved from vector search.
    
    Attributes:
        chunk_id: Unique chunk identifier.
        document_id: Parent document ID.
        content: Chunk text content.
        initial_score: Score from initial retrieval.
        rerank_score: Score after reranking.
        metadata: Additional metadata.
    """
    chunk_id: str
    document_id: str
    content: str
    initial_score: float = 0.0
    rerank_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def final_score(self) -> float:
        """Get final combined score."""
        if self.rerank_score > 0:
            return self.rerank_score
        return self.initial_score


@dataclass
class RAGContext:
    """Context for RAG generation.
    
    Attributes:
        query: Original user query.
        expanded_queries: Query expansion results.
        chunks: Retrieved and ranked chunks.
        compressed_context: Compressed context string.
        metadata: Pipeline metadata.
    """
    query: str
    expanded_queries: List[str] = field(default_factory=list)
    chunks: List[RetrievedChunk] = field(default_factory=list)
    compressed_context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_context_string(self, max_chunks: int = 5) -> str:
        """Get formatted context string for LLM."""
        if self.compressed_context:
            return self.compressed_context
        
        chunks = sorted(self.chunks, key=lambda c: c.final_score, reverse=True)
        chunks = chunks[:max_chunks]
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}] (score: {chunk.final_score:.3f})\n{chunk.content}"
            )
        
        return "\n\n".join(context_parts)


class QueryExpander:
    """Expands queries for better retrieval coverage."""
    
    def __init__(self, llm_provider=None):
        """Initialize query expander.
        
        Args:
            llm_provider: Optional LLM for semantic expansion.
        """
        self._llm = llm_provider
        self._cache: Dict[str, List[str]] = {}
    
    def expand_with_synonyms(self, query: str) -> List[str]:
        """Expand query with synonym variations.
        
        Args:
            query: Original query.
            
        Returns:
            List of expanded queries.
        """
        # Simple rule-based expansion
        expansions = [query]
        
        # Common synonyms map
        synonyms = {
            "show": ["display", "present", "list"],
            "find": ["search", "locate", "get"],
            "explain": ["describe", "elaborate", "clarify"],
            "summary": ["overview", "brief", "synopsis"],
            "details": ["information", "specifics", "data"],
            "create": ["generate", "make", "build"],
        }
        
        words = query.lower().split()
        for word in words:
            if word in synonyms:
                for syn in synonyms[word]:
                    new_query = query.lower().replace(word, syn)
                    if new_query != query.lower():
                        expansions.append(new_query)
        
        return expansions[:5]  # Limit expansions
    
    async def expand_with_llm(self, query: str) -> List[str]:
        """Expand query using LLM for semantic understanding.
        
        Args:
            query: Original query.
            
        Returns:
            List of semantically expanded queries.
        """
        if not self._llm:
            return [query]
        
        # Check cache
        if query in self._cache:
            return self._cache[query]
        
        try:
            prompt = (
                f"Generate 3 alternative phrasings for this search query. "
                f"Return only the queries, one per line.\n\n"
                f"Query: {query}"
            )
            
            response = await self._llm.generate(prompt)
            lines = response.content.strip().split("\n")
            expansions = [query] + [l.strip() for l in lines if l.strip()]
            
            self._cache[query] = expansions[:4]
            return self._cache[query]
            
        except Exception as e:
            logger.warning(f"LLM query expansion failed: {e}")
            return [query]


class CrossEncoderReranker:
    """Reranks results using cross-encoder scoring."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize reranker.
        
        Args:
            model_name: Cross-encoder model name.
        """
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Load the cross-encoder model."""
        if self._initialized:
            return
        
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            self._initialized = True
            logger.info(f"Loaded cross-encoder: {self.model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Reranking will use fallback scoring."
            )
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}")
    
    async def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: Optional[int] = None
    ) -> List[RetrievedChunk]:
        """Rerank chunks using cross-encoder.
        
        Args:
            query: User query.
            chunks: Chunks to rerank.
            top_k: Maximum chunks to return.
            
        Returns:
            Reranked chunks.
        """
        if not chunks:
            return []
        
        if not self._initialized:
            await self.initialize()
        
        if self._model is None:
            # Fallback: use initial scores
            return sorted(chunks, key=lambda c: c.initial_score, reverse=True)
        
        # Create query-document pairs
        pairs = [(query, chunk.content) for chunk in chunks]
        
        try:
            # Get cross-encoder scores
            scores = self._model.predict(pairs)
            
            # Assign rerank scores
            for chunk, score in zip(chunks, scores):
                chunk.rerank_score = float(score)
            
            # Sort by rerank score
            reranked = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)
            
            if top_k:
                reranked = reranked[:top_k]
            
            return reranked
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return sorted(chunks, key=lambda c: c.initial_score, reverse=True)


class ContextCompressor:
    """Compresses retrieved context to fit LLM context window."""
    
    def __init__(
        self,
        max_tokens: int = 2048,
        llm_provider=None
    ):
        """Initialize context compressor.
        
        Args:
            max_tokens: Maximum tokens in compressed context.
            llm_provider: Optional LLM for abstractive compression.
        """
        self.max_tokens = max_tokens
        self._llm = llm_provider
    
    def compress_extractive(
        self,
        chunks: List[RetrievedChunk],
        query: str,
        char_limit: int = 8000
    ) -> str:
        """Compress by extracting most relevant sentences.
        
        Args:
            chunks: Retrieved chunks.
            query: User query for relevance scoring.
            char_limit: Maximum characters.
            
        Returns:
            Compressed context string.
        """
        if not chunks:
            return ""
        
        # Split into sentences and score by query term overlap
        query_terms = set(query.lower().split())
        scored_sentences = []
        
        for chunk in chunks:
            sentences = self._split_sentences(chunk.content)
            for sent in sentences:
                sent_terms = set(sent.lower().split())
                overlap = len(query_terms & sent_terms)
                score = overlap / max(len(query_terms), 1) + chunk.final_score * 0.5
                scored_sentences.append((sent, score, chunk.document_id))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        compressed = []
        current_length = 0
        
        for sent, score, doc_id in scored_sentences:
            if current_length + len(sent) > char_limit:
                break
            compressed.append(sent)
            current_length += len(sent) + 1
        
        return " ".join(compressed)
    
    async def compress_abstractive(
        self,
        chunks: List[RetrievedChunk],
        query: str
    ) -> str:
        """Compress using LLM summarization.
        
        Args:
            chunks: Retrieved chunks.
            query: User query.
            
        Returns:
            Abstractively compressed context.
        """
        if not self._llm or not chunks:
            return self.compress_extractive(chunks, query)
        
        # Combine chunks
        combined = "\n\n".join(c.content for c in chunks[:5])
        
        prompt = (
            f"Summarize the following information to answer: {query}\n\n"
            f"Information:\n{combined}\n\n"
            f"Concise summary:"
        )
        
        try:
            response = await self._llm.generate(prompt)
            return response.content
        except Exception as e:
            logger.warning(f"Abstractive compression failed: {e}")
            return self.compress_extractive(chunks, query)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


class AdvancedRAGPipeline:
    """Advanced RAG pipeline with multi-stage processing.
    
    Pipeline stages:
    1. Query expansion
    2. Multi-query retrieval
    3. Cross-encoder reranking
    4. Context compression
    5. Response generation
    """
    
    def __init__(
        self,
        vector_store=None,
        embedding_service=None,
        llm_provider=None,
        enable_reranking: bool = True,
        enable_compression: bool = True
    ):
        """Initialize RAG pipeline.
        
        Args:
            vector_store: Vector store service.
            embedding_service: Embedding service.
            llm_provider: LLM provider for generation.
            enable_reranking: Enable cross-encoder reranking.
            enable_compression: Enable context compression.
        """
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._llm = llm_provider
        
        self.query_expander = QueryExpander(llm_provider)
        self.reranker = CrossEncoderReranker() if enable_reranking else None
        self.compressor = ContextCompressor(llm_provider=llm_provider) if enable_compression else None
        
        # Configuration
        self.initial_retrieval_k = 20
        self.final_k = 5
    
    async def retrieve(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        expand_query: bool = True
    ) -> RAGContext:
        """Retrieve and process context for a query.
        
        Args:
            query: User query.
            filter_dict: Optional filter for retrieval.
            expand_query: Whether to expand the query.
            
        Returns:
            RAGContext with retrieved chunks.
        """
        context = RAGContext(query=query)
        
        # Stage 1: Query expansion
        if expand_query:
            context.expanded_queries = self.query_expander.expand_with_synonyms(query)
        else:
            context.expanded_queries = [query]
        
        # Stage 2: Multi-query retrieval
        all_chunks = []
        seen_ids = set()
        
        for expanded_query in context.expanded_queries:
            chunks = await self._retrieve_for_query(
                expanded_query,
                self.initial_retrieval_k,
                filter_dict
            )
            
            for chunk in chunks:
                if chunk.chunk_id not in seen_ids:
                    all_chunks.append(chunk)
                    seen_ids.add(chunk.chunk_id)
        
        # Stage 3: Reranking
        if self.reranker and all_chunks:
            all_chunks = await self.reranker.rerank(
                query, all_chunks, self.final_k * 2
            )
        
        context.chunks = all_chunks[:self.final_k]
        
        # Stage 4: Compression (optional)
        if self.compressor and context.chunks:
            context.compressed_context = self.compressor.compress_extractive(
                context.chunks, query
            )
        
        context.metadata = {
            "expanded_queries_count": len(context.expanded_queries),
            "initial_chunks": len(all_chunks),
            "final_chunks": len(context.chunks),
            "reranked": self.reranker is not None,
            "compressed": self.compressor is not None,
        }
        
        return context
    
    async def generate(
        self,
        query: str,
        context: Optional[RAGContext] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate response using RAG context.
        
        Args:
            query: User query.
            context: Pre-retrieved context (or will retrieve).
            system_prompt: Optional system prompt.
            
        Returns:
            Generated response.
        """
        if context is None:
            context = await self.retrieve(query)
        
        if not self._llm:
            return "LLM provider not configured"
        
        context_str = context.get_context_string()
        
        prompt = (
            f"Context:\n{context_str}\n\n"
            f"Question: {query}\n\n"
            f"Answer based on the context provided:"
        )
        
        default_system = (
            "You are a helpful document assistant. Answer questions based on "
            "the provided context. If the context doesn't contain relevant "
            "information, say so."
        )
        
        response = await self._llm.generate(
            prompt,
            system_prompt=system_prompt or default_system
        )
        
        return response.content
    
    async def _retrieve_for_query(
        self,
        query: str,
        top_k: int,
        filter_dict: Optional[Dict[str, Any]]
    ) -> List[RetrievedChunk]:
        """Retrieve chunks for a single query."""
        if not self._vector_store or not self._embedding_service:
            return []
        
        try:
            # Get query embedding
            embedding = await self._embedding_service.embed(query)
            
            # Search vector store
            results = self._vector_store.search(
                embedding,
                top_k=top_k,
                filter_dict=filter_dict
            )
            
            chunks = []
            for result in results:
                chunk = RetrievedChunk(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    content=result.content,
                    initial_score=result.score,
                    metadata=result.metadata
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []


# Global pipeline instance
_rag_pipeline: Optional[AdvancedRAGPipeline] = None


def get_rag_pipeline() -> AdvancedRAGPipeline:
    """Get the global RAG pipeline."""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = AdvancedRAGPipeline()
    return _rag_pipeline


async def create_configured_pipeline(
    vector_store=None,
    embedding_service=None,
    llm_provider=None
) -> AdvancedRAGPipeline:
    """Create a configured RAG pipeline."""
    global _rag_pipeline
    _rag_pipeline = AdvancedRAGPipeline(
        vector_store=vector_store,
        embedding_service=embedding_service,
        llm_provider=llm_provider
    )
    
    # Initialize reranker
    if _rag_pipeline.reranker:
        await _rag_pipeline.reranker.initialize()
    
    return _rag_pipeline
