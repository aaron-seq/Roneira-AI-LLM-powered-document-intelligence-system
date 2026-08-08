"""Cross-encoder reranking of retrieved passages.

Vector search compares a query embedding to passage embeddings that were
computed without ever seeing the query. That is what makes it fast enough to
search a whole corpus, and it is also its ceiling: the passage vector cannot
emphasise the part a particular question is about.

A cross-encoder reads the query and the passage *together* and scores the pair.
It is far too slow to run over a corpus, and exactly right for reordering the
handful of candidates retrieval already narrowed things down to.

The model is ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (~80MB), trained on MS
MARCO passage ranking. It arrives with sentence-transformers, which is already
an optional dependency here, so this adds a download rather than a package.

Like every other optional capability in this codebase, absence is reported
rather than hidden: no sentence-transformers, no download, or a load failure
all leave retrieval working on its existing ranking and say why.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder

    CROSS_ENCODER_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    CrossEncoder = None  # type: ignore[assignment]
    CROSS_ENCODER_AVAILABLE = False

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

#: Passage characters handed to the cross-encoder. The model truncates at 512
#: tokens anyway; cutting here keeps the batch small and predictable.
MAX_PASSAGE_CHARS = 2000


class Reranker:
    """Reorders candidate passages by cross-encoder relevance."""

    def __init__(self, model_name: str = DEFAULT_MODEL, enabled: bool = True):
        self.model_name = model_name
        self._enabled = enabled
        self._model = None
        self._reason: Optional[str] = None if enabled else "reranking is disabled"
        self._initialized = False

    async def initialize(self) -> None:
        """Load the model, degrading to a no-op if it cannot be had."""
        if self._initialized:
            return
        self._initialized = True

        if not self._enabled:
            return

        if not CROSS_ENCODER_AVAILABLE:
            self._degrade(
                "sentence-transformers is not installed, so passages are ranked "
                "by embedding similarity alone"
            )
            return

        try:
            # Loading pulls the model on first use and is CPU-bound; keep it
            # off the event loop, as the embedding service does.
            loop = asyncio.get_running_loop()
            self._model = await loop.run_in_executor(
                None, lambda: CrossEncoder(self.model_name)
            )
            self._reason = None
            logger.info("Reranker initialized with '%s'", self.model_name)
        except Exception as exc:
            self._degrade(f"could not load '{self.model_name}': {exc}")

    def _degrade(self, reason: str) -> None:
        self._model = None
        self._reason = reason
        logger.warning("Reranking unavailable: %s", reason)

    @property
    def is_available(self) -> bool:
        return self._model is not None

    @property
    def degraded_reason(self) -> Optional[str]:
        return self._reason

    async def rank(self, query: str, passages: Sequence[str]) -> Optional[List[int]]:
        """Return candidate indices ordered best-first, or None if unavailable.

        Returning indices rather than reordered passages keeps this decoupled
        from whatever the caller is actually holding, and means the caller
        controls what happens to the scores it already has.
        """
        if not self.is_available or len(passages) < 2:
            return None

        pairs: List[Tuple[str, str]] = [
            (query, passage[:MAX_PASSAGE_CHARS]) for passage in passages
        ]

        try:
            loop = asyncio.get_running_loop()
            scores = await loop.run_in_executor(None, lambda: self._model.predict(pairs))
        except Exception as exc:
            # A reranking failure must not fail the search. The existing order
            # is a worse answer, not no answer.
            logger.warning("Reranking failed, keeping the original order: %s", exc)
            return None

        return sorted(range(len(passages)), key=lambda i: float(scores[i]), reverse=True)
