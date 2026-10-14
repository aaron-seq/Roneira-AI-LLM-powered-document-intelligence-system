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
import importlib
import logging
from typing import Any, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Imported on first use, not at module scope: sentence-transformers pulls in
#: torch and transformers and takes ~4s, which was being paid during ``import
#: backend.main`` — ahead of the listening socket. See the matching note in
#: ``embedding_service``.
_CROSS_ENCODER_AVAILABLE: Optional[bool] = None
_CrossEncoder: Any = None


def _load_cross_encoder() -> bool:
    """Import the cross-encoder, reporting whether it is usable.

    Catches ``Exception`` for the same reason the embedding service does: a
    broken install raises more than ``ImportError``, and an exception escaping
    a background task leaves readiness wedged at 503 rather than degrading.
    """
    global _CROSS_ENCODER_AVAILABLE, _CrossEncoder

    if _CROSS_ENCODER_AVAILABLE is not None:
        return _CROSS_ENCODER_AVAILABLE

    try:
        module = importlib.import_module("sentence_transformers")
        _CrossEncoder = module.CrossEncoder
        _CROSS_ENCODER_AVAILABLE = True
    except ImportError:  # pragma: no cover - optional dependency
        _CROSS_ENCODER_AVAILABLE = False
    except Exception as exc:
        _CROSS_ENCODER_AVAILABLE = False
        logger.error(
            "sentence-transformers is installed but could not be imported: %s: %s",
            type(exc).__name__,
            exc,
            exc_info=True,
        )
    return _CROSS_ENCODER_AVAILABLE


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
        #: Serialises loading, for the same reason the embedding service has
        #: one: startup loads in the background, so a request can arrive
        #: mid-load. The flag alone was not enough — it was set *before* the
        #: download, so a second caller saw "initialized" and went on to rank
        #: with ``_model`` still None, silently skipping reranking.
        self._load_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Load the model, degrading to a no-op if it cannot be had.

        Idempotent, and safe to call concurrently.
        """
        async with self._load_lock:
            if self._initialized:
                return
            await self._load()

    async def _load(self) -> None:
        """Load the cross-encoder, or record why there is not one."""
        self._initialized = True

        if not self._enabled:
            return

        loop = asyncio.get_running_loop()

        # The import is itself expensive (~4s), so it goes to the executor
        # along with the load.
        if not await loop.run_in_executor(None, _load_cross_encoder):
            self._degrade(
                "sentence-transformers is not installed, so passages are ranked "
                "by embedding similarity alone"
            )
            return

        try:
            # Loading pulls the model on first use and is CPU-bound; keep it
            # off the event loop, as the embedding service does.
            self._model = await loop.run_in_executor(
                None, lambda: _CrossEncoder(self.model_name)
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
