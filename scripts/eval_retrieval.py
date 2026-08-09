"""Measure retrieval accuracy against the bundled sample corpus.

Retrieval changes are easy to argue about and hard to settle without numbers.
This indexes `docs/samples/` into a throwaway store, asks a fixed set of
questions whose correct source document is known, and reports how often the
right document comes back and how high it ranks.

    python scripts/eval_retrieval.py                  # current settings
    python scripts/eval_retrieval.py --no-hybrid      # semantic ranking only
    python scripts/eval_retrieval.py --no-rerank      # skip the cross-encoder
    python scripts/eval_retrieval.py --compare        # every combination

The questions are deliberately *paraphrased* rather than quoting the document.
"How much do we owe Quantum Industrial Systems" has to reach "TOTAL DUE";
"vacation days for new starters" has to reach "New Employees (Year 1)". A
keyword index scores badly on these, which is the point — it is what tells you
whether semantic retrieval is actually working rather than coincidentally
matching a shared noun.

Metrics
    recall@k  the correct document appears somewhere in the top k
    MRR       1/rank of the correct document, averaged. Rewards ranking it
              first rather than merely including it, which is what matters
              when only the top few chunks reach the model.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
from typing import List, NamedTuple

# Run from a checkout without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Question(NamedTuple):
    """A question and the sample document that answers it."""

    text: str
    expected: str


#: Every expected answer was read out of the sample it points at. The wording
#: avoids the document's own phrasing wherever possible.
GOLDEN: List[Question] = [
    # --- invoices: numeric and entity lookups ------------------------------
    Question("how much do we owe Quantum Industrial Systems", "INV-2025-1001.pdf"),
    Question("which supplier is based in New York", "INV-2025-1002.pdf"),
    Question(
        "who should I contact at Enterprise Solutions about their bill",
        "INV-2025-1001.pdf",
    ),
    Question("what did we pay for cloud infrastructure setup", "INV-2025-1002.pdf"),
    Question(
        "which invoice covers a backup system and office supplies", "INV-2025-1003.pdf"
    ),
    Question("invoice from the shipping company in Houston", "INV-2025-1003.pdf"),
    Question("INV-2025-1001", "INV-2025-1001.pdf"),
    # --- HR policies: paraphrase-heavy -------------------------------------
    Question("how many vacation days do new starters get", "hr_2024_001.pdf"),
    Question("can I carry unused leave into next year", "hr_2024_001.pdf"),
    Question("what is the allowance for working from home", "hr_2024_002.pdf"),
    Question("how many days a week must I be in the office", "hr_2024_002.pdf"),
    Question("policy for the manufacturing firm in Chicago", "hr_2024_002.pdf"),
    Question(
        "time off entitlement at the San Francisco technology company", "hr_2024_001.pdf"
    ),
]

TOP_K = 5


async def evaluate(hybrid: bool, rerank: bool) -> dict:
    """Index the corpus and score the golden questions."""
    workspace = tempfile.mkdtemp(prefix="roneira-eval-")
    os.environ["VECTOR_STORE_PATH"] = os.path.join(workspace, "vectors")
    os.environ["HYBRID_RETRIEVAL"] = "true" if hybrid else "false"
    os.environ["RERANK_RESULTS"] = "true" if rerank else "false"

    from backend.common.helpers import extract_text_from_file
    from backend.core.config import reload_settings
    from backend.services.retrieval_service import RetrievalService

    reload_settings()
    retrieval = RetrievalService()
    await retrieval.initialize()

    samples_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "samples"
    )

    # Index the *whole* corpus, not just the documents the questions point at.
    # With only the five answer documents present, "is the right one in the top
    # five" is true by construction and the score means nothing. The other
    # seventeen are near-neighbours — more invoices, more HR policies — which
    # is what makes ranking the correct one first a real result.
    indexed = 0
    for filename in sorted(os.listdir(samples_dir)):
        if not filename.lower().endswith((".pdf", ".txt")):
            continue
        try:
            text, _ = await extract_text_from_file(os.path.join(samples_dir, filename))
        except Exception as exc:  # a sample we cannot read is not a test failure
            print(f"  skipped {filename}: {exc}")
            continue
        await retrieval.index_document(
            document_id=filename,
            content=text,
            metadata={"filename": filename},
            owner_id="eval",
        )
        indexed += 1

    hits_at = {1: 0, 3: 0, 5: 0}
    reciprocal = 0.0
    misses: List[str] = []

    for question in GOLDEN:
        result = await retrieval.retrieve(
            query=question.text, top_k=TOP_K, min_score=0.0, owner_id="eval"
        )
        ranked = []
        for item in result.results:
            name = item.metadata.get("filename")
            if name not in ranked:
                ranked.append(name)

        if question.expected in ranked:
            rank = ranked.index(question.expected) + 1
            reciprocal += 1 / rank
            for k in hits_at:
                if rank <= k:
                    hits_at[k] += 1
        else:
            misses.append(question.text)

    # Read before cleanup: the service resets on teardown, and reading after it
    # reported keyword-only matching for a run that used a real model.
    embeddings_are_real = retrieval.embeddings_are_real
    await retrieval.cleanup()

    total = len(GOLDEN)
    return {
        "embeddings_are_real": embeddings_are_real,
        "documents_indexed": indexed,
        "recall@1": hits_at[1] / total,
        "recall@3": hits_at[3] / total,
        "recall@5": hits_at[5] / total,
        "mrr": reciprocal / total,
        "misses": misses,
    }


def _print(label: str, scores: dict) -> None:
    print(
        f"{label:<28} "
        f"recall@1 {scores['recall@1']:.0%}  "
        f"recall@3 {scores['recall@3']:.0%}  "
        f"recall@5 {scores['recall@5']:.0%}  "
        f"MRR {scores['mrr']:.3f}"
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-hybrid", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--compare", action="store_true", help="score every combination")
    args = parser.parse_args()

    print(
        f"{len(GOLDEN)} questions, answers spread over "
        f"{len({q.expected for q in GOLDEN})} documents, scored against the whole "
        f"sample corpus\n"
    )

    if args.compare:
        results = {}
        for hybrid, rerank, label in (
            (False, False, "semantic only"),
            (True, False, "+ keyword (hybrid)"),
            (False, True, "+ cross-encoder"),
            (True, True, "+ both"),
        ):
            results[label] = await evaluate(hybrid=hybrid, rerank=rerank)
            _print(label, results[label])

        if not next(iter(results.values()))["embeddings_are_real"]:
            print(
                "\nWARNING: no embedding model loaded, so these numbers describe "
                "keyword matching. Install sentence-transformers."
            )
        return 0

    scores = await evaluate(hybrid=not args.no_hybrid, rerank=not args.no_rerank)
    _print("current settings", scores)
    if scores["misses"]:
        print("\nNot retrieved in the top 5:")
        for miss in scores["misses"]:
            print(f"  - {miss}")
    if not scores["embeddings_are_real"]:
        print("\nWARNING: keyword-only matching; install sentence-transformers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
