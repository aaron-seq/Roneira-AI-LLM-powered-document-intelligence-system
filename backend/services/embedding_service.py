"""
Embedding Service for Vector Representations

Provides text embedding generation using Sentence Transformers
with support for caching and batch processing.
"""

import logging
import hashlib
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import asyncio
from functools import lru_cache

logger = logging.getLogger(__name__)

# Try to import sentence-transformers, provide fallback
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
    logger.warning(
        "sentence-transformers not installed. "
        "Install with: pip install sentence-transformers"
    )


@dataclass
class EmbeddingResult:
    """Result of an embedding operation."""

    text: str
    embedding: List[float]
    model_name: str
    dimension: int
    text_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "embedding": self.embedding,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "text_hash": self.text_hash,
        }


class EmbeddingCache:
    """Simple in-memory cache for embeddings."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._cache: Dict[str, List[float]] = {}
        self._access_order: List[str] = []

    def _compute_hash(self, text: str, model_name: str) -> str:
        """Compute hash for cache key."""
        content = f"{model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, text: str, model_name: str) -> Optional[List[float]]:
        """Get embedding from cache."""
        key = self._compute_hash(text, model_name)
        if key in self._cache:
            # Move to end of access order (LRU)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def set(self, text: str, model_name: str, embedding: List[float]) -> None:
        """Store embedding in cache."""
        key = self._compute_hash(text, model_name)

        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size and self._access_order:
            oldest_key = self._access_order.pop(0)
            self._cache.pop(oldest_key, None)

        self._cache[key] = embedding
        self._access_order.append(key)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


class EmbeddingService:
    """
    Main service for generating text embeddings.

    Supports multiple embedding models and provides caching
    for improved performance.
    """

    # Popular embedding models with their dimensions
    SUPPORTED_MODELS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-MiniLM-L3-v2": 384,
        "multi-qa-MiniLM-L6-cos-v1": 384,
        "all-distilroberta-v1": 768,
    }

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_size: int = 10000,
        use_cache: bool = True,
    ):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.use_cache = use_cache
        self.cache = EmbeddingCache(max_size=cache_size) if use_cache else None
        self.model: Optional[SentenceTransformer] = None
        self.is_initialized = False
        self._dimension: Optional[int] = None

    async def initialize(self) -> None:
        """Initialize the embedding model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning(
                "Sentence Transformers not available. "
                "Using mock embeddings for development."
            )
            self._dimension = self.SUPPORTED_MODELS.get(self.model_name, 384)
            self.is_initialized = True
            return

        try:
            # Load model in executor to not block event loop
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, lambda: SentenceTransformer(self.model_name)
            )
            self._dimension = self.model.get_sentence_embedding_dimension()
            self.is_initialized = True
            logger.info(
                f"EmbeddingService initialized with model '{self.model_name}' "
                f"(dimension: {self._dimension})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            # Fall back to mock mode
            self._dimension = self.SUPPORTED_MODELS.get(self.model_name, 384)
            self.is_initialized = True

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension or self.SUPPORTED_MODELS.get(self.model_name, 384)

    async def embed_text(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult with the embedding vector
        """
        if not self.is_initialized:
            await self.initialize()

        # Check cache first
        if self.use_cache and self.cache:
            cached = self.cache.get(text, self.model_name)
            if cached is not None:
                return EmbeddingResult(
                    text=text,
                    embedding=cached,
                    model_name=self.model_name,
                    dimension=len(cached),
                    text_hash=hashlib.sha256(text.encode()).hexdigest()[:16],
                )

        # Generate embedding
        embedding = await self._generate_embedding(text)

        # Cache result
        if self.use_cache and self.cache:
            self.cache.set(text, self.model_name, embedding)

        return EmbeddingResult(
            text=text,
            embedding=embedding,
            model_name=self.model_name,
            dimension=len(embedding),
            text_hash=hashlib.sha256(text.encode()).hexdigest()[:16],
        )

    async def embed_texts(
        self, texts: List[str], batch_size: int = 32
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing

        Returns:
            List of EmbeddingResult objects
        """
        if not self.is_initialized:
            await self.initialize()

        results = []
        texts_to_embed = []
        text_indices = []

        # Check cache for each text
        for i, text in enumerate(texts):
            if self.use_cache and self.cache:
                cached = self.cache.get(text, self.model_name)
                if cached is not None:
                    results.append(
                        (
                            i,
                            EmbeddingResult(
                                text=text,
                                embedding=cached,
                                model_name=self.model_name,
                                dimension=len(cached),
                                text_hash=hashlib.sha256(text.encode()).hexdigest()[
                                    :16
                                ],
                            ),
                        )
                    )
                    continue

            texts_to_embed.append(text)
            text_indices.append(i)

        # Batch embed remaining texts
        if texts_to_embed:
            embeddings = await self._generate_embeddings_batch(
                texts_to_embed, batch_size
            )

            for text, embedding, idx in zip(texts_to_embed, embeddings, text_indices):
                # Cache result
                if self.use_cache and self.cache:
                    self.cache.set(text, self.model_name, embedding)

                results.append(
                    (
                        idx,
                        EmbeddingResult(
                            text=text,
                            embedding=embedding,
                            model_name=self.model_name,
                            dimension=len(embedding),
                            text_hash=hashlib.sha256(text.encode()).hexdigest()[:16],
                        ),
                    )
                )

        # Sort by original index and extract results
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        if self.model is None:
            # Mock embedding for development
            return self._generate_mock_embedding(text)

        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, lambda: self.model.encode(text, convert_to_numpy=True)
        )
        return embedding.tolist()

    async def _generate_embeddings_batch(
        self, texts: List[str], batch_size: int
    ) -> List[List[float]]:
        """Generate embeddings for batch of texts."""
        if self.model is None:
            # Mock embeddings for development
            return [self._generate_mock_embedding(t) for t in texts]

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
        )
        return [e.tolist() for e in embeddings]

    def _generate_mock_embedding(self, text: str) -> List[float]:
        """Generate deterministic mock embedding for development."""
        import hashlib

        hash_bytes = hashlib.sha256(text.encode()).digest()

        # Generate deterministic floats from hash
        embedding = []
        for i in range(self.dimension):
            byte_idx = i % len(hash_bytes)
            value = (hash_bytes[byte_idx] - 128) / 128.0
            embedding.append(value)

        return embedding

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.cache:
            return {"caching_enabled": False}

        return {
            "caching_enabled": True,
            "cache_size": self.cache.size,
            "max_cache_size": self.cache.max_size,
            "model_name": self.model_name,
            "dimension": self.dimension,
        }

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.cache:
            self.cache.clear()
        self.model = None
        self.is_initialized = False
        logger.info("EmbeddingService cleaned up")
