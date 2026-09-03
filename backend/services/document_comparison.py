"""Comparing two documents.

"What changed between v1 and v2 of this contract" is the question this
answers. The result is derived entirely from the two documents' extracted
text — no model is involved, so there is nothing to hallucinate and nothing to
degrade when Ollama is absent. Every reported change quotes the text it came
from and the page it is on, which is the same standard the rest of the system
holds citations to.

Paragraphs are the unit of comparison rather than lines or words. A line diff
of reflowed text reports every line of a paragraph as changed when one word
moved; a word diff loses the context needed to see what a change means.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.common.helpers import PAGE_MARKER
from backend.services.retrieval_service import _page_for_offset, _page_offsets

#: Changes returned per comparison. A rewritten hundred-page contract would
#: otherwise produce a response nobody can read and a payload nobody wants.
MAX_CHANGES = 200

#: Paragraphs shorter than this are joined onto the previous one. Extraction
#: leaves stray fragments — a page number, a heading rule — and reporting each
#: as an independent change buries the real edits.
MIN_PARAGRAPH_CHARS = 3

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_WHITESPACE = re.compile(r"\s+")


@dataclass
class Paragraph:
    """One unit of comparison, with the page it appears on."""

    text: str
    page: Optional[int]


@dataclass
class Change:
    """One difference between the two documents."""

    #: "added", "removed" or "changed".
    kind: str
    left: List[str] = field(default_factory=list)
    right: List[str] = field(default_factory=list)
    left_page: Optional[int] = None
    right_page: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "left": self.left,
            "right": self.right,
            "left_page": self.left_page,
            "right_page": self.right_page,
        }


def _normalise(text: str) -> str:
    """Collapse whitespace so reflowed text is not reported as a change."""
    return _WHITESPACE.sub(" ", text).strip()


def split_paragraphs(content: str) -> List[Paragraph]:
    """Split extracted text into paragraphs, each tagged with its page.

    ``content`` is the stored text, still carrying its ``--- Page N ---``
    markers; those drive the page numbers and are not themselves compared.
    """
    offsets = _page_offsets(content)
    paragraphs: List[Paragraph] = []

    position = 0
    for block in _PARAGRAPH_SPLIT.split(content):
        start = content.find(block, position)
        if start == -1:
            start = position
        position = start + len(block)

        # Page markers are positioning devices, not content.
        body = _normalise(PAGE_MARKER.sub("", block))
        if len(body) < MIN_PARAGRAPH_CHARS:
            continue

        paragraphs.append(Paragraph(text=body, page=_page_for_offset(offsets, start)))

    return paragraphs


def compare_documents(left_content: str, right_content: str) -> Dict[str, Any]:
    """Return the paragraph-level differences between two documents.

    ``left`` is the older or baseline document; ``right`` is what it is being
    compared against.
    """
    left = split_paragraphs(left_content)
    right = split_paragraphs(right_content)

    matcher = difflib.SequenceMatcher(
        a=[p.text for p in left], b=[p.text for p in right], autojunk=False
    )

    changes: List[Change] = []
    unchanged = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            unchanged += i2 - i1
            continue

        kind = {"replace": "changed", "delete": "removed", "insert": "added"}[tag]
        changes.append(
            Change(
                kind=kind,
                left=[p.text for p in left[i1:i2]],
                right=[p.text for p in right[j1:j2]],
                left_page=left[i1].page if i1 < len(left) else None,
                right_page=right[j1].page if j1 < len(right) else None,
            )
        )

    truncated = len(changes) > MAX_CHANGES
    if truncated:
        changes = changes[:MAX_CHANGES]

    total = len(left) + len(right)
    return {
        "changes": [change.to_dict() for change in changes],
        "added": sum(1 for c in changes if c.kind == "added"),
        "removed": sum(1 for c in changes if c.kind == "removed"),
        "changed": sum(1 for c in changes if c.kind == "changed"),
        "unchanged_paragraphs": unchanged,
        "left_paragraphs": len(left),
        "right_paragraphs": len(right),
        # 1.0 means the two texts are paragraph-for-paragraph identical.
        "similarity": round((2 * unchanged / total) if total else 1.0, 4),
        "truncated": truncated,
    }
