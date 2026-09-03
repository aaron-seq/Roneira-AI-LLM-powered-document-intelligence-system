"""Entities, personal data and extractive summaries.

All three are derived from the text with patterns and word statistics, so they
work when no LLM is reachable — the ordinary case for a local run — and cannot
report something the document does not contain.
"""

from __future__ import annotations

import time

import pytest

from backend.services.document_insights import (
    build_insights,
    detect_pii,
    extract_entities,
    summarise,
)

INVOICE = (
    "ACME Corporation Invoice INV-2025-1001 issued 2025-03-01.\n"
    "Billed to Contoso Ltd, accounts@contoso.example.\n"
    "Total amount due: $12,480.00, a 15% increase on the previous quarter.\n"
    "Questions: call 555-123-4567 or see https://acme.example/invoices\n"
)


class TestEntityExtraction:
    def test_finds_the_entities_an_invoice_actually_contains(self):
        entities = extract_entities(INVOICE)

        assert "accounts@contoso.example" in entities["EMAIL"]
        assert any("12,480" in m for m in entities["MONEY"])
        assert "15%" in entities["PERCENT"]
        assert any("acme.example" in u for u in entities["URL"])

    def test_entities_are_deduplicated(self):
        repeated = "Write to a@b.example. Again: a@b.example. And a@b.example."

        assert extract_entities(repeated)["EMAIL"] == ["a@b.example"]

    def test_a_document_with_no_entities_yields_nothing_invented(self):
        assert extract_entities("The quick brown fox jumped over it.") == {}


class TestPIIDetection:
    SENSITIVE = (
        "Employee record. SSN: 123-45-6789. Card 4111111111111111.\n"
        "Contact: person@example.com, 555-987-6543.\n"
    )

    def test_detects_personal_data(self):
        report = detect_pii(self.SENSITIVE)

        assert report["found"] > 0
        assert "SSN" in report["by_type"]
        assert "EMAIL" in report["by_type"]

    def test_the_report_never_carries_the_pii_itself(self):
        """This result is stored on the document and returned by the API.

        The underlying PIIMatch.to_dict() includes the matched text; echoing it
        back would put national insurance and card numbers in the database and
        in API responses, which is worse than not detecting them.
        """
        report = detect_pii(self.SENSITIVE)
        serialised = repr(report)

        assert "123-45-6789" not in serialised
        assert "4111111111111111" not in serialised
        assert "person@example.com" not in serialised

    def test_positions_are_reported_so_a_ui_can_highlight(self):
        report = detect_pii(self.SENSITIVE)

        assert report["positions"]
        assert all({"start", "end", "type"} <= set(p) for p in report["positions"])

    def test_a_clean_document_reports_nothing(self):
        report = detect_pii("Revenue grew. The team shipped the migration.")

        assert report["found"] == 0
        assert report["by_type"] == {}


class TestExtractiveSummary:
    def test_the_summary_uses_only_the_documents_own_sentences(self):
        text = (
            "The quarterly revenue reached four point two million dollars. "
            "The engineering team completed the platform migration early. "
            "Customer churn fell for the third consecutive quarter. "
            "The office coffee machine was replaced in March. "
            "Revenue growth was driven by enterprise contract renewals."
        )

        summary = summarise(text, sentences=2)

        assert summary
        for sentence in summary.split(". "):
            fragment = sentence.strip().rstrip(".")
            if fragment:
                assert fragment in text, "summaries must not invent wording"

    def test_short_text_is_returned_rather_than_mangled(self):
        assert summarise("One short line about revenue.") is not None

    def test_page_markers_do_not_reach_the_summary(self):
        """``--- Page N ---`` is a retrieval device, not content.

        The extractor writes these separators so a chunk can be mapped back to
        a page and cited. Retrieval and comparison both strip them; the
        summary path did not, so every summary shown in the UI and every
        Markdown export opened with "--- Page 1 ---".
        """
        paginated = (
            "--- Page 1 ---\n"
            "The quarterly revenue reached four point two million dollars.\n"
            "--- Page 2 ---\n"
            "Customer churn fell for the third consecutive quarter.\n"
        )

        summary = summarise(paginated, sentences=2)

        assert "--- Page" not in summary
        assert "revenue" in summary or "churn" in summary

    def test_a_short_paginated_document_is_still_cleaned(self):
        """The short-text path returns the text as-is; it must still be clean.

        This is the case the bug actually showed up in: a one-page invoice has
        too few sentences to rank, so the summariser hands the text straight
        back — markers included.
        """
        assert "--- Page" not in summarise("--- Page 1 ---\nTotal due: $8,385.00.")


class TestInsightsShape:
    def test_build_insights_returns_both_sections(self):
        insights = build_insights(INVOICE)

        assert set(insights) == {"entities", "pii"}


def _upload_and_wait(client, headers, name, body):
    response = client.post(
        "/api/documents/upload",
        headers=headers,
        files={"file": (name, body.encode(), "text/plain")},
    )
    assert response.status_code == 202, response.text
    document_id = response.json()["document_id"]

    deadline = time.time() + 30
    while time.time() < deadline:
        status = client.get(
            f"/api/documents/{document_id}/status", headers=headers
        ).json()
        if status["status"] in ("completed", "failed"):
            assert status["status"] == "completed", status
            return document_id
        time.sleep(0.05)
    pytest.fail("document did not finish processing")


class TestInsightsReachTheDocument:
    """The point of wiring these in: they show up on the processed document."""

    def test_a_processed_document_carries_entities_and_a_summary(
        self, client, auth_headers
    ):
        document_id = _upload_and_wait(
            client, auth_headers, "insights_invoice.txt", INVOICE
        )

        body = client.get(f"/api/documents/{document_id}", headers=auth_headers).json()
        insights = body["ai_insights"]

        assert insights["entities"]["EMAIL"] == ["accounts@contoso.example"]
        assert insights["pii"]["found"] >= 1
        # No LLM in the test environment, so the summary is the extractive one.
        assert insights["summary"]
        assert insights["summary_source"] in ("extractive", "llm")
