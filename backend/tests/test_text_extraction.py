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
from backend.common.ocr import ocr_availability, ocr_is_available

#: OCR needs the tesseract *binary*, which is a system package. Where it is
#: absent these tests skip rather than fail — but CI installs it, so the
#: scanned-document path is genuinely exercised there.
requires_ocr = pytest.mark.skipif(
    not ocr_is_available(),
    reason=f"tesseract is not available: {ocr_availability()[1]}",
)


def _render_scan(lines, size=(1600, 700)):
    """Build a page image the way a scanner would: black text, no text layer."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", size, "white")
    y = 60
    for line in lines:
        # The default PIL font is small; draw then upscale so tesseract sees
        # glyphs at a realistic size.
        strip = Image.new("RGB", (900, 40), "white")
        ImageDraw.Draw(strip).text((5, 10), line, fill="black")
        image.paste(strip.resize((1400, 62)), (80, y))
        y += 110
    return image


SCAN_LINES = [
    "ACME CORPORATION INVOICE",
    "Invoice Number: INV-2025-1001",
    "Total Amount Due: 12480.00 USD",
]


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


class TestOCR:
    """Scanned documents must become searchable, and say that they were OCR'd.

    Before this, an image-only PDF produced no text and was rejected, and an
    image upload was refused outright — the largest gap between what the
    README promised and what the pipeline did.
    """

    def test_availability_is_reported_with_a_reason(self):
        """Degraded modes are explicit here, as with embeddings and the LLM."""
        available, reason = ocr_availability()
        assert isinstance(available, bool)
        assert reason, "an unavailable engine must say why"

    @requires_ocr
    @pytest.mark.asyncio
    async def test_reads_an_image_only_pdf(self, tmp_path):
        """A real scan: rasterised text, no text layer whatsoever."""
        path = tmp_path / "scanned.pdf"
        _render_scan(SCAN_LINES).save(str(path), "PDF", resolution=200.0)

        text, metadata = await extract_text_from_file(str(path))

        assert "ACME" in text.upper()
        assert "12480" in text
        assert metadata["ocr_pages"] == [1], "the OCR'd page must be recorded"
        assert metadata["ocr_engine"]

    @requires_ocr
    @pytest.mark.asyncio
    async def test_ocr_text_still_carries_page_markers(self, tmp_path):
        """Without the marker an OCR'd page cannot be cited."""
        path = tmp_path / "scanned.pdf"
        _render_scan(SCAN_LINES).save(str(path), "PDF", resolution=200.0)

        text, _ = await extract_text_from_file(str(path))

        assert "--- Page 1 ---" in text

    @requires_ocr
    @pytest.mark.asyncio
    async def test_reads_an_uploaded_image(self, tmp_path):
        path = tmp_path / "receipt.png"
        _render_scan(SCAN_LINES).save(str(path))

        text, metadata = await extract_text_from_file(str(path))

        assert "ACME" in text.upper()
        assert metadata["ocr_pages"] == [1]
        assert metadata["pages"] == 1

    @pytest.mark.asyncio
    async def test_a_pdf_with_a_text_layer_is_not_ocred(
        self, tmp_path, minimal_pdf_bytes
    ):
        """OCR is the expensive path; it must not run when text is present."""
        path = tmp_path / "native.pdf"
        path.write_bytes(minimal_pdf_bytes)

        text, metadata = await extract_text_from_file(str(path))

        assert "Zephyr" in text
        assert "ocr_pages" not in metadata

    @pytest.mark.asyncio
    async def test_a_scan_without_ocr_explains_itself(self, tmp_path, monkeypatch):
        """With OCR off, the failure names the reason instead of being generic."""
        import backend.common.helpers as helpers

        monkeypatch.setattr(
            helpers,
            "ocr_availability",
            lambda: (False, "OCR is disabled (ENABLE_OCR=false)."),
        )
        path = tmp_path / "receipt.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)

        with pytest.raises(TextExtractionError) as excinfo:
            await extract_text_from_file(str(path))

        assert "ENABLE_OCR" in str(excinfo.value)


class TestExtractionFailsLoudly:
    @pytest.mark.asyncio
    async def test_unsupported_extension_raises(self, tmp_path):
        path = tmp_path / "archive.zip"
        path.write_bytes(b"PK\x03\x04 not a document")

        with pytest.raises(TextExtractionError, match="not a supported"):
            await extract_text_from_file(str(path))

    @pytest.mark.asyncio
    async def test_an_unreadable_image_is_rejected_rather_than_faked(self, tmp_path):
        """The old code returned a placeholder string and indexed it.

        This file is not a decodable image, so it fails whether or not OCR is
        installed — but it must fail with a reason, never as empty content.
        """
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
