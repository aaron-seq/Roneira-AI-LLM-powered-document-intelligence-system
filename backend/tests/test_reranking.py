"""Cross-encoder reranking.

Measured over `docs/samples/` with scripts/eval_retrieval.py, reranking moves
recall@1 from 69% to 77% and recall@3 from 92% to 100%. These tests cover the
behaviour that has to hold regardless of the model: it must never lose results,
never rewrite the scores a citation reports, and never turn its own failure
into a failed search.
"""

from __future__ import annotations

import pytest

from backend.services.reranking import Reranker

#: The model is an ~80MB download. Where it is absent these skip rather than
#: fail, exactly as the OCR tests do.
_probe = Reranker()


async def _available() -> bool:
    await _probe.initialize()
    return _probe.is_available


class TestAvailabilityIsHonest:
    @pytest.mark.asyncio
    async def test_a_disabled_reranker_says_so_and_does_nothing(self):
        reranker = Reranker(enabled=False)
        await reranker.initialize()

        assert reranker.is_available is False
        assert reranker.degraded_reason
        assert await reranker.rank("anything", ["a", "b"]) is None

    @pytest.mark.asyncio
    async def test_an_unloadable_model_degrades_instead_of_raising(self):
        """A missing model must not take the search down with it."""
        reranker = Reranker(model_name="not-a-real-model/does-not-exist")
        await reranker.initialize()

        assert reranker.is_available is False
        assert reranker.degraded_reason
        assert await reranker.rank("q", ["a", "b"]) is None

    @pytest.mark.asyncio
    async def test_a_single_candidate_needs_no_reranking(self):
        reranker = Reranker()
        await reranker.initialize()

        assert await reranker.rank("q", ["only one"]) is None


class TestRanking:
    @pytest.mark.asyncio
    async def test_it_ranks_the_passage_that_answers_the_question_first(self):
        if not await _available():
            pytest.skip(f"cross-encoder unavailable: {_probe.degraded_reason}")

        passages = [
            "The office cafeteria serves lunch between noon and two.",
            "Employees in their first year receive 15 days of paid time off.",
            "The car park is accessible with a staff badge.",
        ]

        order = await _probe.rank("how much annual leave do new joiners get", passages)

        assert order is not None
        assert order[0] == 1, "the PTO passage answers the question"

    @pytest.mark.asyncio
    async def test_every_candidate_survives_reranking(self):
        """Reordering must not silently drop a result."""
        if not await _available():
            pytest.skip(f"cross-encoder unavailable: {_probe.degraded_reason}")

        passages = [f"passage number {i} about invoices" for i in range(6)]

        order = await _probe.rank("invoice total", passages)

        assert order is not None
        assert sorted(order) == list(range(len(passages)))


class TestRetrievalIntegration:
    @pytest.mark.asyncio
    async def test_scores_still_mean_what_they_meant(self):
        """Cross-encoder outputs are unbounded logits, not similarities.

        Writing them into `score` would silently break RETRIEVAL_MIN_SCORE and
        the "% match" shown next to a citation, so reranking may reorder
        results but must leave their scores alone.
        """
        from backend.services.vector_store_service import SearchResult

        before = [
            SearchResult(
                chunk_id=f"c{i}",
                document_id="d",
                content=f"passage {i}",
                score=0.9 - i / 10,
                metadata={},
            )
            for i in range(4)
        ]
        original = {r.chunk_id: r.score for r in before}

        reranker = Reranker()
        await reranker.initialize()
        order = await reranker.rank("anything at all", [r.content for r in before])

        after = [before[i] for i in order] if order else before
        assert {r.chunk_id: r.score for r in after} == original
