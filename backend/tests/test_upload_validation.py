"""Upload validation.

Uploads previously trusted the filename extension and buffered the entire file
into memory before checking its size, so a renamed binary was accepted as a
"PDF" and any caller could force an arbitrary allocation.
"""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from backend.services.file_validation import (
    FileValidationError,
    detect_content_type,
    normalise_extension,
    save_upload_stream,
    validate_filename,
    verify_content_matches_extension,
)


class TestFilenameValidation:
    def test_accepts_a_supported_extension(self):
        assert validate_filename("report.pdf") == ".pdf"

    def test_extension_check_is_case_insensitive(self):
        assert validate_filename("REPORT.PDF") == ".pdf"

    def test_rejects_an_unsupported_extension(self):
        with pytest.raises(FileValidationError, match="Unsupported file type"):
            validate_filename("payload.exe")

    def test_rejects_a_missing_extension(self):
        with pytest.raises(FileValidationError):
            validate_filename("README")

    def test_rejects_an_empty_name(self):
        with pytest.raises(FileValidationError, match="No filename"):
            validate_filename("")

    @pytest.mark.parametrize(
        "name", ["../../etc/passwd.txt", "dir/file.txt", "back\\slash.txt"]
    )
    def test_rejects_path_separators(self, name):
        with pytest.raises(FileValidationError, match="path separators"):
            validate_filename(name)

    def test_rejects_a_null_byte(self):
        with pytest.raises(FileValidationError):
            validate_filename("evil\x00.txt")

    def test_double_extension_is_judged_by_the_last_one(self):
        """`invoice.pdf.exe` is an executable, not a PDF."""
        assert normalise_extension("invoice.pdf.exe") == ".exe"
        with pytest.raises(FileValidationError):
            validate_filename("invoice.pdf.exe")


class TestContentDetection:
    @pytest.mark.parametrize(
        "head,extension,expected",
        [
            (b"%PDF-1.7\n...", ".pdf", "application/pdf"),
            (b"\x89PNG\r\n\x1a\n", ".png", "image/png"),
            (b"\xff\xd8\xff\xe0", ".jpg", "image/jpeg"),
            (b"Plain readable text", ".txt", "text/plain"),
        ],
    )
    def test_detects_type_from_leading_bytes(self, head, extension, expected):
        assert detect_content_type(head, extension) == expected

    def test_an_inconclusive_libmagic_answer_falls_back_to_signatures(self, monkeypatch):
        """Detection must not depend on which host the service runs on.

        libmagic ships on most Linux images and rarely on Windows. It returns
        `application/octet-stream` for a file whose signature is recognisable
        but whose body is truncated, so treating that as a verdict meant the
        same upload was accepted on one host and 400'd on the other.
        """
        from backend.services import file_validation

        monkeypatch.setattr(file_validation, "_MAGIC_AVAILABLE", True)
        monkeypatch.setattr(
            file_validation,
            "_magic",
            SimpleNamespace(from_buffer=lambda *a, **k: "application/octet-stream"),
        )

        assert detect_content_type(b"\x89PNG\r\n\x1a\n", ".png") == "image/png"
        assert detect_content_type(b"%PDF-1.7\n...", ".pdf") == "application/pdf"

    def test_a_specific_libmagic_answer_still_wins(self, monkeypatch):
        """Falling back must not blunt the check that catches renamed files."""
        from backend.services import file_validation

        monkeypatch.setattr(file_validation, "_MAGIC_AVAILABLE", True)
        monkeypatch.setattr(
            file_validation,
            "_magic",
            SimpleNamespace(from_buffer=lambda *a, **k: "application/x-dosexec"),
        )

        # Executable bytes wearing a .pdf extension are still refused.
        detected = detect_content_type(b"MZ\x90\x00", ".pdf")
        assert detected == "application/x-dosexec"
        with pytest.raises(FileValidationError, match="does not match"):
            verify_content_matches_extension(detected, ".pdf")

    def test_matching_content_and_extension_is_accepted(self):
        verify_content_matches_extension("application/pdf", ".pdf")

    def test_binary_content_in_a_txt_file_is_rejected(self):
        with pytest.raises(FileValidationError, match="contents are binary"):
            verify_content_matches_extension("application/octet-stream", ".txt")

    def test_a_renamed_executable_is_rejected(self):
        """The whole point: extension says PDF, bytes say otherwise."""
        detected = detect_content_type(b"\x7fELF\x02\x01\x01\x00", ".pdf")
        with pytest.raises(FileValidationError, match="does not match"):
            verify_content_matches_extension(detected, ".pdf")


class _FakeUpload:
    """Minimal stand-in for Starlette's UploadFile."""

    def __init__(self, filename: str, data: bytes):
        self.filename = filename
        self._stream = io.BytesIO(data)

    async def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)


class TestStreamingUpload:
    @pytest.mark.asyncio
    async def test_writes_the_file_and_reports_its_checksum(self, tmp_path):
        import hashlib

        payload = b"Quarterly revenue report for ACME Corporation."
        destination = str(tmp_path / "out.txt")

        result = await save_upload_stream(
            _FakeUpload("report.txt", payload), destination, max_bytes=1024
        )

        assert result.size_bytes == len(payload)
        assert result.checksum == hashlib.sha256(payload).hexdigest()
        with open(destination, "rb") as handle:
            assert handle.read() == payload

    @pytest.mark.asyncio
    async def test_rejects_a_file_over_the_limit(self, tmp_path):
        destination = str(tmp_path / "big.txt")
        oversized = b"A" * 5000

        with pytest.raises(FileValidationError) as excinfo:
            await save_upload_stream(
                _FakeUpload("big.txt", oversized), destination, max_bytes=1024
            )
        assert excinfo.value.status_code == 413

    @pytest.mark.asyncio
    async def test_partial_file_is_removed_after_a_rejection(self, tmp_path):
        """A rejected upload must not leave bytes behind on disk."""
        import os

        destination = str(tmp_path / "partial.txt")
        with pytest.raises(FileValidationError):
            await save_upload_stream(
                _FakeUpload("partial.txt", b"B" * 5000), destination, max_bytes=64
            )
        assert not os.path.exists(destination)

    @pytest.mark.asyncio
    async def test_rejects_an_empty_file(self, tmp_path):
        with pytest.raises(FileValidationError, match="empty"):
            await save_upload_stream(
                _FakeUpload("empty.txt", b""), str(tmp_path / "empty.txt"), 1024
            )

    @pytest.mark.asyncio
    async def test_rejects_content_that_contradicts_the_extension(self, tmp_path):
        with pytest.raises(FileValidationError, match="does not match"):
            await save_upload_stream(
                _FakeUpload("fake.pdf", b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 100),
                str(tmp_path / "fake.pdf"),
                max_bytes=4096,
            )


class TestUploadEndpointRejections:
    """The same rules, enforced through the HTTP API."""

    def test_rejects_an_unsupported_extension(self, client, auth_headers):
        response = client.post(
            "/api/documents/upload",
            headers=auth_headers,
            files={"file": ("virus.exe", b"MZ\x90\x00binary", "application/exe")},
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_rejects_a_renamed_binary(self, client, auth_headers):
        response = client.post(
            "/api/documents/upload",
            headers=auth_headers,
            files={
                "file": ("report.pdf", b"\x7fELF\x02\x01\x01\x00" * 20, "application/pdf")
            },
        )
        assert response.status_code == 400
        assert "does not match" in response.json()["detail"]

    def test_rejects_an_empty_file(self, client, auth_headers):
        response = client.post(
            "/api/documents/upload",
            headers=auth_headers,
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert response.status_code == 400

    def test_oversized_upload_returns_413(self, client, app, auth_headers, monkeypatch):
        # Patch the settings object the running processor actually holds, not
        # whatever get_settings() returns now — services capture their config
        # at construction time.
        processor = app.state.container.document_processor
        monkeypatch.setattr(processor.settings, "max_file_size", 512, raising=False)

        response = client.post(
            "/api/documents/upload",
            headers=auth_headers,
            files={"file": ("large.txt", b"X" * 4096, "text/plain")},
        )
        assert response.status_code == 413

    def test_supported_formats_are_discoverable(self, client):
        """Clients should not have to guess what the server accepts."""
        response = client.get("/api/documents/formats/supported")
        assert response.status_code == 200
        body = response.json()
        assert ".pdf" in body["extensions"]
        assert body["max_file_size_mb"] > 0
