"""Structured insights derived from a document's text, without a model.

Three things a reader wants that the LLM was previously the only source of:
what entities appear, whether the document carries personal data, and what it
says in brief. All three are computed from the text with regular expressions
and word statistics, so they are available when Ollama is not — which is the
common case for a local run — and they cannot hallucinate.

This module is a thin, deliberate adapter over the pattern extractors in
``entity_extraction_service`` and ``pii_detection_service``. It exists rather
than calling those services directly for one reason worth stating:

    ``PIIDetectionService.generate_report`` embeds the matched text itself
    (``PIIMatch.to_dict`` has an ``original`` field). Persisting that would
    write detected national insurance and card numbers into the database and
    hand them back over the API — worse than not detecting them at all.

So the report built here carries types, counts, confidence and positions, and
never the matched value.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from backend.common.helpers import strip_page_markers
from backend.services.entity_extraction_service import EntityType, PatternEntityExtractor
from backend.services.pii_detection_service import PIIDetector
from backend.services.summarization_service import ExtractiveSummarizer

logger = logging.getLogger(__name__)

#: Entities kept per type. A hundred-page contract yields hundreds of dates;
#: the first few are what a person scanning the document actually reads.
MAX_ENTITIES_PER_TYPE = 25

#: Text handed to the extractors. The regex passes are linear but repeated per
#: pattern, and beyond this the marginal entity is not worth the wait on a
#: background worker shared with every other upload.
MAX_ANALYSIS_CHARS = 200_000

#: Below this confidence a PII hit is noise — a nine-digit invoice number
#: reads as a national insurance number. Kept in the report but flagged, so
#: the count a user sees is not dominated by false positives.
PII_CONFIDENCE_FLOOR = 0.8

_entity_extractor = PatternEntityExtractor()
_pii_detector = PIIDetector()
_summariser = ExtractiveSummarizer()


def extract_entities(text: str) -> Dict[str, List[str]]:
    """Group normalised entities by type, most frequent first."""
    entities = _entity_extractor.extract(text[:MAX_ANALYSIS_CHARS])

    counts: Dict[EntityType, Dict[str, int]] = {}
    for entity in entities:
        by_value = counts.setdefault(entity.entity_type, {})
        by_value[entity.normalized] = by_value.get(entity.normalized, 0) + 1

    grouped: Dict[str, List[str]] = {}
    for entity_type, by_value in counts.items():
        ranked = sorted(by_value.items(), key=lambda item: (-item[1], item[0]))
        grouped[entity_type.value] = [
            value for value, _ in ranked[:MAX_ENTITIES_PER_TYPE]
        ]
    return grouped


def detect_pii(text: str) -> Dict[str, Any]:
    """Summarise personal data found in the text.

    Returns counts and positions only. The matched values are deliberately
    absent: this result is persisted on the document and returned by the API,
    and echoing the PII back would defeat the point of detecting it.
    """
    matches = _pii_detector.detect(text[:MAX_ANALYSIS_CHARS])

    by_type: Dict[str, int] = {}
    confident = 0
    for match in matches:
        by_type[match.pii_type.value] = by_type.get(match.pii_type.value, 0) + 1
        if match.confidence >= PII_CONFIDENCE_FLOOR:
            confident += 1

    return {
        "found": len(matches),
        "confident": confident,
        "by_type": by_type,
        # Offsets let a UI highlight the spans without ever transporting them.
        "positions": [
            {"start": m.start_pos, "end": m.end_pos, "type": m.pii_type.value}
            for m in matches
            if m.confidence >= PII_CONFIDENCE_FLOOR
        ][:200],
    }


def summarise(text: str, sentences: int = 3) -> str:
    """Pick the most representative sentences from the text.

    Extractive, so every word is the document's own. Used when the LLM cannot
    be reached; an abstractive summary reads better but is not available then,
    and inventing one is not an option.

    The ``--- Page N ---`` separators are stripped first. They are a retrieval
    positioning device, and leaving them in put "--- Page 1 ---" at the top of
    every summary shown in the UI and every Markdown export.
    """
    try:
        # Returns a plain string, despite the SummaryResult dataclass defined
        # alongside it — that is only used by the abstractive path.
        return _summariser.summarize(
            strip_page_markers(text)[:MAX_ANALYSIS_CHARS], num_sentences=sentences
        ).strip()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Extractive summarisation failed: %s", exc)
        return ""


def build_insights(text: str) -> Dict[str, Any]:
    """Everything derivable from the text alone."""
    return {
        "entities": extract_entities(text),
        "pii": detect_pii(text),
    }
