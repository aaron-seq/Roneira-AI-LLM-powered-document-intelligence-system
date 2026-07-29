"""Text extraction, and what happens when it fails.

Extraction errors used to be *returned as the document's text*: a corrupt or
unsupported file was recorded as `completed` with the body
"Error extracting text: ...". That string was then chunked, embedded, indexed,
and would surface as a search hit and be cited in an answer as though it were
document content. A failure must fail.
"""

from __future__ import annotations

import time

import pytest

from backend.common.helpers import TextExtractionError, extract_text_from_file


def _wait(client, headers, document_id, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/documents/{document_id}/status", headers=headers).json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.05)
    pytest.fail("document did not finish processing")


class TestExtractionSucceeds:
    @pytest.mark.asyncio
    async def test_reads_a_plain_text_file(self, tmp_path):
        path = tmp_path / "note.txt"
        path.write_text("Quarterly revenue was 4.2 million dollars.")

        text, metadata = await extract_text_from_file(str(path))

        assert "Quarterly revenue" in text
        assert metadata["word_count"] == 6
        assert metadata["file_type"] == ".txt"

    @pytest.mark.asyncio
    async def test_falls_back_when_utf8_decoding_fails(self, tmp_path):
        """One bad byte should not cost the whole document."""
        path = tmp_path / "legacy.txt"
        path.write_bytes(b"Caf\xe9 receipt total 12.00")

        text, _ = await extract_text_from_file(str(path))
        assert "receipt total" in text

    @pytest.mark.asyncio
    async def test_pdf_pages_are_marked_for_citation(self, tmp_path, minimal_pdf_bytes):
        path = tmp_path / "report.pdf"
        path.write_bytes(minimal_pdf_bytes)

        text, metadata = await extract_text_from_file(str(path))

        assert "--- Page 1 ---" in text, "page markers drive citation page numbers"
        assert metadata["pages"] == 1


class TestExtractionFailsLoudly:
    @pytest.mark.asyncio
    async def test_unsupported_extension_raises(self, tmp_path):
        path = tmp_path / "archive.zip"
        path.write_bytes(b"PK\x03\x04 not a document")

        with pytest.raises(TextExtractionError, match="not a supported"):
            await extract_text_from_file(str(path))

    @pytest.mark.asyncio
    async def test_an_image_is_rejected_rather_than_faked(self, tmp_path):
        """The old code returned a placeholder string and indexed it."""
        path = tmp_path / "scan.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)

        with pytest.raises(TextExtractionError) as excinfo:
            await extract_text_from_file(str(path))

        message = str(excinfo.value)
        assert "OCR" in message, "the message should say why, not just fail"

    @pytest.mark.asyncio
    async def test_an_empty_document_raises(self, tmp_path):
        path = tmp_path / "blank.txt"
        path.write_text("   \n\n  ")

        with pytest.raises(TextExtractionError, match="No text could be read"):
            await extract_text_from_file(str(path))

    @pytest.mark.asyncio
    async def test_a_corrupt_pdf_raises(self, tmp_path):
        path = tmp_path / "broken.pdf"
        path.write_bytes(b"%PDF-1.4\nthis is not actually a pdf structure")

        with pytest.raises(TextExtractionError):
            await extract_text_from_file(str(path))

    @pytest.mark.asyncio
    async def test_failure_messages_never_masquerade_as_content(self, tmp_path):
        """The regression itself: no error text may be returned as document text."""
        path = tmp_path / "archive.zip"
        path.write_bytes(b"PK\x03\x04")

        try:
            text, _ = await extract_text_from_file(str(path))
        except TextExtractionError:
            return  # correct behaviour

        pytest.fail(
            f"extraction returned text instead of raising; this string would "
            f"have been indexed and cited: {text[:80]!r}"
        )


class TestFailedDocumentsAreNotSearchable:
    """End-to-end: a document that fails extraction must not enter the index."""

    def test_an_unreadable_document_is_marked_failed(self, client, auth_headers):
        response = client.post(
            "/api/documents/upload",
            headers=auth_headers,
            # A valid PNG by content type, so upload validation accepts it;
            # extraction is what must reject it.
            files={
                "file": ("scan.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 128, "image/png")
            },
        )
        assert response.status_code == 202
        document_id = response.json()["document_id"]

        final = _wait(client, auth_headers, document_id)
        assert final["status"] == "failed"
        assert final["chunk_count"] == 0

    def test_a_failed_document_explains_why(self, client, auth_headers):
        response = client.post(
            "/api/documents/upload",
            headers=auth_headers,
            files={
                "file": ("photo.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 128, "image/png")
            },
        )
        final = _wait(client, auth_headers, response.json()["document_id"])

        assert final["status"] == "failed"
        assert final.get("error"), "a failed document must say what went wrong"
        assert "OCR" in final["error"]

    def test_the_failure_text_is_not_retrievable(self, client, auth_headers):
        """Searching for the error message must not return the failed document."""
        response = client.post(
            "/api/documents/upload",
            headers=auth_headers,
            files={
                "file": ("img.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 128, "image/png")
            },
        )
        document_id = response.json()["document_id"]
        _wait(client, auth_headers, document_id)

        search = client.post(
            "/api/search",
            headers=auth_headers,
            json={"query": "OCR not enabled image", "top_k": 10, "min_score": 0.0},
        )
        returned = {r["document_id"] for r in search.json()["results"]}
        assert document_id not in returned
