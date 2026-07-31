"""Exporting a document's analysis.

An export is the most likely thing to be pasted into a ticket or attached to
an email, which makes it the worst place to leak a card number. The rule the
rest of the API follows holds here: counts, never values.
"""

from __future__ import annotations

import time

import pytest

from backend.services.document_export import (
    render_comparison_markdown,
    render_markdown,
)

DOCUMENT = {
    "document_id": "abc123",
    "filename": "invoice.pdf",
    "page_count": 2,
    "word_count": 180,
    "checksum": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
    "embeddings_are_real": True,
    "ai_insights": {
        "summary": "ACME billed Contoso 12,480.00 for the migration.",
        "summary_source": "extractive",
        "entities": {"EMAIL": ["accounts@contoso.example"], "MONEY": ["$12,480.00"]},
        "pii": {"found": 2, "confident": 2, "by_type": {"SSN": 1, "EMAIL": 1}},
    },
    "result": {
        "original_text": "ACME Corporation Invoice INV-2025-1001.",
        "metadata": {"ocr_pages": [2]},
    },
}


class TestMarkdownExport:
    def test_includes_the_headline_facts(self):
        markdown = render_markdown(DOCUMENT)

        assert markdown.startswith("# invoice.pdf")
        assert "2 pages" in markdown
        assert "180 words" in markdown

    def test_labels_an_extractive_summary_as_the_documents_own_words(self):
        markdown = render_markdown(DOCUMENT)

        assert "no model was involved" in markdown
        assert "ACME billed Contoso" in markdown

    def test_flags_ocr_pages_so_the_reader_can_weigh_them(self):
        markdown = render_markdown(DOCUMENT)

        assert "Page(s) 2" in markdown
        assert "OCR" in markdown

    def test_reports_pii_counts_but_never_values(self):
        """An export is the worst possible place to carry a card number."""
        document = {
            **DOCUMENT,
            "ai_insights": {
                **DOCUMENT["ai_insights"],
                "pii": {"found": 1, "confident": 1, "by_type": {"SSN": 1}},
            },
            "result": {
                "original_text": "Nothing sensitive here.",
                "metadata": {},
            },
        }

        markdown = render_markdown(document)

        assert "1 x ssn" in markdown
        assert "123-45-6789" not in markdown

    def test_a_keyword_only_index_is_disclosed(self):
        markdown = render_markdown({**DOCUMENT, "embeddings_are_real": False})

        assert "keywords" in markdown

    def test_a_bare_document_still_renders(self):
        assert render_markdown({"document_id": "x"}).startswith("# x")


class TestComparisonExport:
    COMPARISON = {
        "left_filename": "v1.txt",
        "right_filename": "v2.txt",
        "similarity": 0.75,
        "changed": 1,
        "added": 1,
        "removed": 0,
        "changes": [
            {
                "kind": "changed",
                "left": ["Payment terms are net 30 days."],
                "right": ["Payment terms are net 45 days."],
                "left_page": 1,
                "right_page": 1,
            }
        ],
        "truncated": False,
    }

    def test_renders_both_sides_of_a_change(self):
        markdown = render_comparison_markdown(self.COMPARISON)

        assert "v1.txt → v2.txt" in markdown
        assert "75% unchanged" in markdown
        assert "~~Payment terms are net 30 days.~~" in markdown
        assert "- Payment terms are net 45 days." in markdown

    def test_says_the_result_is_not_generated(self):
        assert "nothing here is generated" in render_comparison_markdown(self.COMPARISON)


def _upload(client, headers, name, body):
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


class TestExportEndpoint:
    BODY = (
        "ACME Corporation Invoice INV-2025-1001.\n\n"
        "Contact accounts@contoso.example about the $12,480.00 balance.\n"
    )

    def test_exports_markdown_as_a_download(self, client, auth_headers):
        document_id = _upload(client, auth_headers, "export_me.txt", self.BODY)

        response = client.get(
            f"/api/documents/{document_id}/export", headers=auth_headers
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/markdown")
        assert "attachment" in response.headers["content-disposition"]
        assert response.text.startswith("# export_me.txt")

    def test_another_users_document_cannot_be_exported(
        self, client, admin_headers, auth_headers
    ):
        document_id = _upload(client, auth_headers, "private.txt", self.BODY)

        response = client.get(
            f"/api/documents/{document_id}/export", headers=admin_headers
        )

        assert response.status_code == 404

    def test_export_requires_authentication(self, client):
        assert client.get("/api/documents/anything/export").status_code == 401

    def test_comparison_exports_markdown(self, client, auth_headers):
        left = _upload(client, auth_headers, "cmp_a.txt", self.BODY)
        right = _upload(client, auth_headers, "cmp_b.txt", self.BODY + "\nExtra.\n")

        response = client.get(
            "/api/documents/compare",
            headers=auth_headers,
            params={"left": left, "right": right, "fmt": "markdown"},
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/markdown")
        assert "cmp_a.txt" in response.text

    def test_an_unknown_format_is_rejected(self, client, auth_headers):
        response = client.get(
            "/api/documents/compare",
            headers=auth_headers,
            params={"left": "a", "right": "b", "fmt": "pdf"},
        )

        assert response.status_code == 422
