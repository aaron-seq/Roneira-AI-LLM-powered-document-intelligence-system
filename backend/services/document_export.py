"""Rendering a processed document as something portable.

The point of exporting is to take the work out of the tool: a summary, the
extracted details and the provenance, in a file that can be pasted into a
ticket or attached to an email. Markdown because it reads as plain text when
nothing renders it, and because every wiki and issue tracker accepts it.

What is deliberately absent: the detected personal data. The export carries
the counts, so a reader knows the document was flagged, and never the values —
the same rule the API follows, for the same reason.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _lines_for_entities(entities: Dict[str, List[str]]) -> List[str]:
    if not entities:
        return []
    lines = ["## Extracted details", ""]
    for entity_type, values in sorted(entities.items()):
        lines.append(f"- **{entity_type.title()}**: {', '.join(values)}")
    lines.append("")
    return lines


def _lines_for_pii(pii: Dict[str, Any]) -> List[str]:
    if not pii or not pii.get("found"):
        return []
    counts = ", ".join(
        f"{count} x {kind.lower().replace('_', ' ')}"
        for kind, count in sorted(pii.get("by_type", {}).items())
    )
    return [
        "## Possible personal data",
        "",
        f"{counts}.",
        "",
        "> Found by pattern matching, so treat this as a prompt to check rather",
        "> than a verdict. The values themselves are deliberately not included",
        "> in this export.",
        "",
    ]


def render_markdown(document: Dict[str, Any]) -> str:
    """Render a document's analysis as Markdown.

    ``document`` is the payload of ``GET /api/documents/{id}``.
    """
    insights = document.get("ai_insights") or {}
    result = document.get("result") or {}
    metadata = result.get("metadata") or {}

    title = document.get("filename") or document.get("document_id") or "Document"
    lines: List[str] = [f"# {title}", ""]

    facts = []
    if document.get("page_count"):
        facts.append(f"{document['page_count']} pages")
    if document.get("word_count"):
        facts.append(f"{document['word_count']} words")
    if document.get("checksum"):
        facts.append(f"SHA-256 `{document['checksum'][:16]}…`")
    if facts:
        lines += [" · ".join(facts), ""]

    # Provenance a reader needs in order to weigh the rest of the file.
    caveats = []
    if metadata.get("ocr_pages"):
        pages = ", ".join(str(p) for p in metadata["ocr_pages"])
        caveats.append(
            f"Page(s) {pages} had no text layer and were read with OCR, so that "
            f"text is only as good as the scan."
        )
    if document.get("embeddings_are_real") is False:
        caveats.append(
            "Indexed without an embedding model, so it is matched on keywords "
            "rather than meaning."
        )
    if caveats:
        lines += ["> " + c for c in caveats] + [""]

    if insights.get("summary"):
        source = insights.get("summary_source")
        lines += ["## Summary", ""]
        if source == "extractive":
            lines += [
                "_Sentences selected from the document itself; no model was involved._",
                "",
            ]
        lines += [insights["summary"], ""]

    lines += _lines_for_entities(insights.get("entities") or {})
    lines += _lines_for_pii(insights.get("pii") or {})

    text = result.get("original_text") or ""
    if text.strip():
        lines += ["## Extracted text", "", "```", text.strip(), "```", ""]

    return "\n".join(lines).rstrip() + "\n"


def render_comparison_markdown(comparison: Dict[str, Any]) -> str:
    """Render a document comparison as Markdown."""
    left = comparison.get("left_filename") or comparison.get("left_document_id")
    right = comparison.get("right_filename") or comparison.get("right_document_id")

    similarity = round((comparison.get("similarity") or 0) * 100)
    lines = [
        f"# {left} → {right}",
        "",
        f"{similarity}% unchanged · {comparison.get('changed', 0)} changed · "
        f"{comparison.get('added', 0)} added · {comparison.get('removed', 0)} removed",
        "",
        "_Computed from the text of both documents; nothing here is generated._",
        "",
    ]

    for change in comparison.get("changes", []):
        page = change.get("left_page") or change.get("right_page")
        heading = change["kind"].title() + (f" (page {page})" if page else "")
        lines += [f"## {heading}", ""]
        for paragraph in change.get("left", []):
            lines.append(f"- ~~{paragraph}~~")
        for paragraph in change.get("right", []):
            lines.append(f"- {paragraph}")
        lines.append("")

    if comparison.get("truncated"):
        lines += ["_Only the first changes are shown._", ""]

    return "\n".join(lines).rstrip() + "\n"
