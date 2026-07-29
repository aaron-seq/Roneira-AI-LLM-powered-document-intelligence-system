"""
Embedding Service for Vector Representations

Provides text embedding generation using Sentence Transformers
with support for caching and batch processing.
"""

import asyncio
import hashlib
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Word-ish tokens for the lexical fallback vectorizer.
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

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


class EmbeddingModelUnavailable(RuntimeError):
    """Raised when a real embedding model is required but cannot be loaded."""


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
        require_real_model: bool = False,
    ):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.use_cache = use_cache
        self.cache = EmbeddingCache(max_size=cache_size) if use_cache else None
        self.model: Optional[SentenceTransformer] = None
        self.is_initialized = False
        self._dimension: Optional[int] = None
        #: When True, initialization raises instead of degrading to the
        #: deterministic fallback. Deployments that serve real answers should
        #: set this: pseudo-embeddings return confident nonsense.
        self.require_real_model = require_real_model
        self._degraded_reason: Optional[str] = None

    async def initialize(self) -> None:
        """Load the embedding model.

        Raises:
            EmbeddingModelUnavailable: if the model cannot be loaded and
                ``require_real_model`` is set.
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self._degrade(
                "sentence-transformers is not installed "
                "(pip install sentence-transformers)"
            )
            return

        try:
            # Load in an executor: model loading is CPU-bound and would
            # otherwise stall the event loop for seconds during startup.
            loop = asyncio.get_running_loop()
            self.model = await loop.run_in_executor(
                None, lambda: SentenceTransformer(self.model_name)
            )
            self._dimension = self.model.get_sentence_embedding_dimension()
            self.is_initialized = True
            self._degraded_reason = None
            logger.info(
                "EmbeddingService initialized with model '%s' (dimension: %s)",
                self.model_name,
                self._dimension,
            )
        except Exception as exc:
            self._degrade(f"failed to load model '{self.model_name}': {exc}")

    def _degrade(self, reason: str) -> None:
        """Enter (or refuse to enter) the deterministic fallback mode."""
        if self.require_real_model:
            raise EmbeddingModelUnavailable(
                f"Semantic embeddings are required but unavailable: {reason}. "
                "Install the model, or set REQUIRE_REAL_EMBEDDINGS=false to "
                "fall back to keyword-only lexical matching."
            )

        self._degraded_reason = reason
        self._dimension = self.SUPPORTED_MODELS.get(self.model_name, 384)
        self.is_initialized = True
        logger.warning(
            "EmbeddingService is using the LEXICAL FALLBACK (%s). Search will "
            "match on keywords only; paraphrases and synonyms will not be "
            "found. Install sentence-transformers for semantic search.",
            reason,
        )

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension or self.SUPPORTED_MODELS.get(self.model_name, 384)

    @property
    def is_real(self) -> bool:
        """True when embeddings come from an actual model."""
        return self.model is not None

    @property
    def backend(self) -> str:
        """Identifier for the active embedding backend."""
        return "sentence-transformers" if self.is_real else "lexical-fallback"

    @property
    def degraded_reason(self) -> Optional[str]:
        """Why the fallback is active, or ``None`` when embeddings are real."""
        return self._degraded_reason

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
                                text_hash=hashlib.sha256(text.encode()).hexdigest()[:16],
                            ),
                        )
                    )
                    continue

            texts_to_embed.append(text)
            text_indices.append(i)

        # Batch embed remaining texts
        if texts_to_embed:
            embeddings = await self._generate_embeddings_batch(texts_to_embed, batch_size)

            for text, embedding, idx in zip(
                texts_to_embed, embeddings, text_indices, strict=True
            ):
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
        """Lexical fallback vector, used when no embedding model is loaded.

        This is the hashing trick: tokens are hashed into the vector's
        dimensions with sub-linear term frequency weighting, then L2
        normalised. The result supports genuine **keyword** matching — a query
        sharing words with a chunk scores highly — but carries no semantic
        understanding, so paraphrases and synonyms will not match.

        Why not random vectors: the original implementation tiled 32 hash bytes
        across every dimension, so unrelated texts correlated arbitrarily and
        search returned noise. Pure pseudo-random vectors are no better —
        cosine similarity between them is ~0, so nothing is ever retrieved.
        A hashed bag-of-words keeps local development and CI meaningful without
        a multi-gigabyte model download, while still being reported honestly as
        a non-semantic backend.
        """
        dimension = self.dimension
        vector = [0.0] * dimension

        tokens = _TOKEN_PATTERN.findall(text.lower())
        if not tokens:
            return vector

        counts: Dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

        for token, count in counts.items():
            # Not a security primitive: sha1 is used here purely to spread tokens
            # deterministically across the vector's dimensions.
            digest = hashlib.sha1(token.encode("utf-8"), usedforsecurity=False).digest()
            index = int.from_bytes(digest[:4], "big") % dimension
            # Sign bit from a different digest byte spreads collisions in both
            # directions instead of letting them always accumulate.
            sign = 1.0 if digest[4] & 1 else -1.0
            # Sub-linear scaling: a word repeated 20 times is not 20x as
            # important as one appearing once.
            vector[index] += sign * (1.0 + math.log(count))

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0.0:
            return vector
        return [value / magnitude for value in vector]

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get embedding backend and cache statistics."""
        stats: Dict[str, Any] = {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "backend": self.backend,
            "embeddings_are_real": self.is_real,
            "caching_enabled": bool(self.cache),
        }
        if self._degraded_reason:
            stats["degraded_reason"] = self._degraded_reason
        if self.cache:
            stats["cache_size"] = self.cache.size
            stats["max_cache_size"] = self.cache.max_size
        return stats

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.cache:
            self.cache.clear()
        self.model = None
        self.is_initialized = False
        logger.info("EmbeddingService cleaned up")
