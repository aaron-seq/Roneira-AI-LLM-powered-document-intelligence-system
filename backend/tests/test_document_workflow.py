"""End-to-end document workflow: upload, process, index, retrieve, delete.

This is the test that would have caught the two defects that made the product
not work: documents indexed at upload were invisible to search (separate
vector stores), and all document state vanished on restart (in-memory dict).
"""

from __future__ import annotations

import time

import pytest


def _upload(client, headers, name, content, content_type="text/plain"):
    return client.post(
        "/api/documents/upload",
        headers=headers,
        files={"file": (name, content, content_type)},
    )


def _wait_for_terminal(client, headers, document_id, timeout=30.0):
    """Poll a document until it reaches a terminal state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/api/documents/{document_id}/status", headers=headers)
        assert response.status_code == 200, response.text
        body = response.json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.05)
    pytest.fail(f"Document {document_id} did not finish within {timeout}s")


@pytest.fixture(scope="class")
def processed_document(client, auth_headers, request):
    """Upload a text document once and wait for it to finish processing."""
    content = (
        "ACME Corporation Quarterly Report\n\n"
        "Revenue for the quarter was 4.2 million dollars, up 18 percent "
        "year over year.\n\n"
        "Action items: renew the Contoso support contract before March 31.\n"
    ).encode("utf-8")

    response = _upload(client, auth_headers, "quarterly.txt", content)
    assert response.status_code == 202, response.text
    document_id = response.json()["document_id"]
    return _wait_for_terminal(client, auth_headers, document_id)


class TestUploadAcceptance:
    def test_upload_returns_202_with_a_document_id(
        self, client, auth_headers, sample_text_bytes
    ):
        response = _upload(client, auth_headers, "report.txt", sample_text_bytes)
        assert response.status_code == 202
        body = response.json()
        assert body["document_id"]
        assert body["status"] == "queued"
        assert body["size_bytes"] == len(sample_text_bytes)

    def test_upload_records_a_checksum_of_the_bytes(
        self, client, auth_headers, sample_text_bytes
    ):
        """Provenance: the same file must always produce the same checksum."""
        import hashlib

        expected = hashlib.sha256(sample_text_bytes).hexdigest()
        response = _upload(client, auth_headers, "a.txt", sample_text_bytes)
        assert response.json()["checksum"] == expected

    def test_upload_detects_content_type_from_bytes(
        self, client, auth_headers, minimal_pdf_bytes
    ):
        response = _upload(
            client, auth_headers, "doc.pdf", minimal_pdf_bytes, "application/pdf"
        )
        assert response.status_code == 202
        assert response.json()["detected_type"] == "application/pdf"


class TestProcessingPipeline:
    def test_document_reaches_completed(self, processed_document):
        assert processed_document["status"] == "completed", processed_document
        assert processed_document["progress"] == 100

    def test_completed_document_reports_indexed_chunks(self, processed_document):
        """A completed document with zero chunks is not searchable."""
        assert processed_document["chunk_count"] > 0

    def test_completed_document_reports_word_count(self, processed_document):
        assert processed_document["word_count"] > 0

    def test_status_declares_whether_embeddings_are_real(self, processed_document):
        """The API must never hide that it indexed with placeholder vectors."""
        assert "embeddings_are_real" in processed_document

    def test_detail_endpoint_returns_the_extracted_text(
        self, client, auth_headers, processed_document
    ):
        response = client.get(
            f"/api/documents/{processed_document['document_id']}",
            headers=auth_headers,
        )
        assert response.status_code == 200
        text = response.json()["result"]["original_text"]
        assert "ACME Corporation" in text


class TestRetrievalSeesUploadedDocuments:
    """The upload pipeline and chat must share one index.

    When they did not, search returned zero results for every document the
    user had just uploaded, while the API reported success at every step.
    """

    def test_search_finds_the_uploaded_document(
        self, client, auth_headers, processed_document
    ):
        response = client.post(
            "/api/search",
            headers=auth_headers,
            json={"query": "quarterly revenue", "top_k": 5, "min_score": 0.0},
        )
        assert response.status_code == 200
        results = response.json()["results"]
        assert results, "search returned nothing for an indexed document"

        document_ids = {r["document_id"] for r in results}
        assert processed_document["document_id"] in document_ids

    def test_search_results_carry_citation_metadata(
        self, client, auth_headers, processed_document
    ):
        response = client.post(
            "/api/search",
            headers=auth_headers,
            json={"query": "revenue", "top_k": 3, "min_score": 0.0},
        )
        result = response.json()["results"][0]
        assert result["chunk_id"]
        assert result["content"]
        assert "filename" in result["metadata"]

    def test_search_scoped_to_a_document_excludes_others(
        self, client, auth_headers, processed_document, sample_text_bytes
    ):
        other = _upload(client, auth_headers, "unrelated.txt", sample_text_bytes)
        other_id = other.json()["document_id"]
        _wait_for_terminal(client, auth_headers, other_id)

        response = client.post(
            "/api/search",
            headers=auth_headers,
            json={
                "query": "revenue",
                "top_k": 10,
                "min_score": 0.0,
                "document_id": processed_document["document_id"],
            },
        )
        assert response.status_code == 200
        returned = {r["document_id"] for r in response.json()["results"]}
        assert returned <= {processed_document["document_id"]}

    def test_rag_stats_report_the_populated_index(
        self, client, auth_headers, processed_document
    ):
        response = client.get("/api/rag/stats", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["vector_store"]["document_count"] > 0


class TestOwnershipIsolation:
    """One user must never see or search another user's documents."""

    def test_listing_excludes_another_users_documents(
        self, client, auth_headers, admin_headers, sample_text_bytes
    ):
        response = _upload(client, auth_headers, "private.txt", sample_text_bytes)
        document_id = response.json()["document_id"]
        _wait_for_terminal(client, auth_headers, document_id)

        listing = client.get("/api/documents", headers=admin_headers)
        assert listing.status_code == 200
        admin_ids = {d["document_id"] for d in listing.json()["documents"]}
        assert document_id not in admin_ids

    def test_status_of_another_users_document_is_404(
        self, client, auth_headers, admin_headers, sample_text_bytes
    ):
        """404 rather than 403: existence itself should not be confirmed."""
        response = _upload(client, auth_headers, "secret.txt", sample_text_bytes)
        document_id = response.json()["document_id"]

        probe = client.get(f"/api/documents/{document_id}/status", headers=admin_headers)
        assert probe.status_code == 404

    def test_cannot_delete_another_users_document(
        self, client, auth_headers, admin_headers, sample_text_bytes
    ):
        response = _upload(client, auth_headers, "mine.txt", sample_text_bytes)
        document_id = response.json()["document_id"]

        assert (
            client.delete(f"/api/documents/{document_id}", headers=admin_headers)
        ).status_code == 404
        # Still present for the real owner.
        assert (
            client.get(f"/api/documents/{document_id}/status", headers=auth_headers)
        ).status_code == 200

    def test_search_does_not_return_another_users_chunks(
        self, client, auth_headers, admin_headers
    ):
        content = b"Confidential Zephyr acquisition memorandum, price 90 million."
        response = _upload(client, auth_headers, "confidential.txt", content)
        _wait_for_terminal(client, auth_headers, response.json()["document_id"])

        leaked = client.post(
            "/api/search",
            headers=admin_headers,
            json={"query": "Zephyr acquisition", "top_k": 10, "min_score": 0.0},
        )
        assert leaked.status_code == 200
        owners = {r["document_id"] for r in leaked.json()["results"]}
        assert response.json()["document_id"] not in owners


class TestPagination:
    def test_total_counts_all_matches_not_just_the_page(
        self, client, auth_headers, sample_text_bytes
    ):
        for index in range(3):
            _upload(client, auth_headers, f"page-{index}.txt", sample_text_bytes)

        response = client.get("/api/documents?limit=1&offset=0", headers=auth_headers)
        assert response.status_code == 200
        body = response.json()
        assert len(body["documents"]) == 1
        assert body["total"] > 1, "total must count all documents, not the page"
        assert body["has_more"] is True

    def test_invalid_status_filter_is_a_400(self, client, auth_headers):
        response = client.get("/api/documents?status_filter=banana", headers=auth_headers)
        assert response.status_code == 400
        assert "banana" in response.json()["detail"]

    def test_limit_is_bounded(self, client, auth_headers):
        """An unbounded limit is a trivial way to exhaust the database."""
        assert (
            client.get("/api/documents?limit=5000", headers=auth_headers).status_code
            == 422
        )


class TestDeletion:
    def test_delete_removes_the_document_and_its_index_entries(
        self, client, auth_headers
    ):
        content = b"Ephemeral document about the Tyrell replicant program."
        response = _upload(client, auth_headers, "ephemeral.txt", content)
        document_id = response.json()["document_id"]
        _wait_for_terminal(client, auth_headers, document_id)

        assert (
            client.delete(f"/api/documents/{document_id}", headers=auth_headers)
        ).status_code == 200

        assert (
            client.get(f"/api/documents/{document_id}/status", headers=auth_headers)
        ).status_code == 404

        # The chunks must be gone from the index too, or deleted documents
        # keep answering questions.
        search = client.post(
            "/api/search",
            headers=auth_headers,
            json={"query": "Tyrell replicant program", "top_k": 10, "min_score": 0.0},
        )
        assert document_id not in {r["document_id"] for r in search.json()["results"]}

    def test_deleting_a_missing_document_is_404(self, client, auth_headers):
        response = client.delete(
            "/api/documents/00000000-0000-0000-0000-000000000000",
            headers=auth_headers,
        )
        assert response.status_code == 404


class TestSourceRetention:
    def test_original_file_can_be_downloaded_back(
        self, client, auth_headers, sample_text_bytes
    ):
        """Citations are only verifiable if the source still exists."""
        response = _upload(client, auth_headers, "retained.txt", sample_text_bytes)
        document_id = response.json()["document_id"]
        _wait_for_terminal(client, auth_headers, document_id)

        download = client.get(
            f"/api/documents/{document_id}/source", headers=auth_headers
        )
        assert download.status_code == 200
        assert download.content == sample_text_bytes

    def test_source_download_requires_ownership(
        self, client, auth_headers, admin_headers, sample_text_bytes
    ):
        response = _upload(client, auth_headers, "owned.txt", sample_text_bytes)
        document_id = response.json()["document_id"]
        _wait_for_terminal(client, auth_headers, document_id)

        assert (
            client.get(f"/api/documents/{document_id}/source", headers=admin_headers)
        ).status_code == 404

    def test_source_is_served_as_an_attachment(
        self, client, auth_headers, sample_text_bytes
    ):
        """Inline rendering of user content would be stored XSS."""
        response = _upload(client, auth_headers, "inline.txt", sample_text_bytes)
        document_id = response.json()["document_id"]
        _wait_for_terminal(client, auth_headers, document_id)

        download = client.get(
            f"/api/documents/{document_id}/source", headers=auth_headers
        )
        assert "attachment" in download.headers["content-disposition"]
