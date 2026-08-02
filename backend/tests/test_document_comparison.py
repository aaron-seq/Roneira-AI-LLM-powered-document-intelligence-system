"""Comparing two documents.

The result is derived from the stored text of both documents, so unlike chat
it does not depend on the LLM and cannot report a change that is not there.
"""

from __future__ import annotations

import time

import pytest

from backend.services.document_comparison import compare_documents, split_paragraphs

CONTRACT_V1 = (
    "--- Page 1 ---\n"
    "Master Services Agreement between ACME Corporation and Contoso Ltd.\n\n"
    "Payment terms are net 30 days from invoice date.\n\n"
    "Either party may terminate with 90 days written notice.\n"
)

CONTRACT_V2 = (
    "--- Page 1 ---\n"
    "Master Services Agreement between ACME Corporation and Contoso Ltd.\n\n"
    "Payment terms are net 45 days from invoice date.\n\n"
    "Either party may terminate with 90 days written notice.\n\n"
    "Late payments accrue interest at 1.5 percent per month.\n"
)


class TestParagraphSplitting:
    def test_page_markers_become_page_numbers_not_content(self):
        paragraphs = split_paragraphs(CONTRACT_V1)

        assert all("--- Page" not in p.text for p in paragraphs)
        assert all(p.page == 1 for p in paragraphs)

    def test_reflowed_whitespace_is_not_a_difference(self):
        """A paragraph rewrapped at a different width has not changed."""
        wrapped = "Payment terms are\nnet 30 days\nfrom invoice date."
        flat = "Payment terms are net 30 days from invoice date."

        assert split_paragraphs(wrapped)[0].text == split_paragraphs(flat)[0].text

    def test_pages_are_tracked_across_a_multi_page_document(self):
        content = (
            "--- Page 1 ---\nFirst page body text here.\n\n"
            "--- Page 2 ---\nSecond page body text here.\n"
        )
        pages = [p.page for p in split_paragraphs(content)]
        assert pages == [1, 2]


class TestComparison:
    def test_an_identical_document_reports_no_changes(self):
        result = compare_documents(CONTRACT_V1, CONTRACT_V1)

        assert result["changes"] == []
        assert result["similarity"] == 1.0

    def test_a_modified_clause_is_reported_as_changed(self):
        result = compare_documents(CONTRACT_V1, CONTRACT_V2)

        changed = [c for c in result["changes"] if c["kind"] == "changed"]
        assert changed, "the altered payment term should be reported"
        assert "net 30 days" in changed[0]["left"][0]
        assert "net 45 days" in changed[0]["right"][0]

    def test_an_added_clause_is_reported_as_added(self):
        result = compare_documents(CONTRACT_V1, CONTRACT_V2)

        added = [c for c in result["changes"] if c["kind"] == "added"]
        assert added
        assert "interest" in added[0]["right"][0]

    def test_a_removed_clause_is_reported_as_removed(self):
        result = compare_documents(CONTRACT_V2, CONTRACT_V1)

        removed = [c for c in result["changes"] if c["kind"] == "removed"]
        assert removed
        assert "interest" in removed[0]["left"][0]

    def test_changes_carry_the_page_they_are_on(self):
        """A change nobody can locate in the original is not much use."""
        result = compare_documents(CONTRACT_V1, CONTRACT_V2)

        assert all(c["left_page"] == 1 or c["right_page"] == 1 for c in result["changes"])

    def test_unchanged_paragraphs_are_counted(self):
        result = compare_documents(CONTRACT_V1, CONTRACT_V2)

        # The title and the termination clause are untouched.
        assert result["unchanged_paragraphs"] == 2

    def test_similarity_falls_when_documents_diverge(self):
        same = compare_documents(CONTRACT_V1, CONTRACT_V1)["similarity"]
        different = compare_documents(CONTRACT_V1, "Something else entirely here.")[
            "similarity"
        ]

        assert same == 1.0
        assert different < same

    def test_comparing_empty_text_does_not_explode(self):
        result = compare_documents("", "")
        assert result["similarity"] == 1.0
        assert result["changes"] == []


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
        status_body = client.get(
            f"/api/documents/{document_id}/status", headers=headers
        ).json()
        if status_body["status"] in ("completed", "failed"):
            assert status_body["status"] == "completed", status_body
            return document_id
        time.sleep(0.05)
    pytest.fail("document did not finish processing")


class TestComparisonEndpoint:
    def test_compares_two_owned_documents(self, client, auth_headers):
        left = _upload(client, auth_headers, "contract_v1.txt", CONTRACT_V1)
        right = _upload(client, auth_headers, "contract_v2.txt", CONTRACT_V2)

        response = client.get(
            "/api/documents/compare",
            headers=auth_headers,
            params={"left": left, "right": right},
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["left_filename"] == "contract_v1.txt"
        assert body["changes"], "the documents differ"
        assert body["similarity"] < 1.0

    def test_the_literal_path_is_not_read_as_a_document_id(self, client, auth_headers):
        """`/compare` must not be matched by `/{document_id}`.

        FastAPI resolves routes in declaration order, so registering this after
        the parameterised route would look up a document called "compare" and
        always 404.
        """
        response = client.get(
            "/api/documents/compare",
            headers=auth_headers,
            params={"left": "missing-a", "right": "missing-b"},
        )

        assert response.status_code == 404
        assert "missing-a" in response.json()["detail"]

    def test_another_users_document_cannot_be_compared(
        self, client, auth_headers, admin_headers
    ):
        """Diffing is a read; it must respect the same ownership boundary.

        Both documents are uploaded by the demo user and the comparison is
        attempted as admin. Uploading one *as* admin would test the same
        boundary, but it would also leave the admin account holding an indexed
        document, and two chat tests depend on it having none.
        """
        left = _upload(client, auth_headers, "left.txt", CONTRACT_V1)
        right = _upload(client, auth_headers, "right.txt", CONTRACT_V2)

        response = client.get(
            "/api/documents/compare",
            headers=admin_headers,
            params={"left": left, "right": right},
        )

        # 404, not 403: confirming the id exists would leak it.
        assert response.status_code == 404

    def test_requires_authentication(self, client):
        response = client.get(
            "/api/documents/compare", params={"left": "a", "right": "b"}
        )
        assert response.status_code == 401
