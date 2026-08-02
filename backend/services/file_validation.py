"""Upload validation.

The previous upload path trusted the filename extension, then read the whole
file into memory *before* checking its size. That accepts a renamed executable
as a "PDF" and lets an unauthenticated caller allocate arbitrary memory.

This module validates in the order that actually protects the server:

1. Extension is on the allow-list (cheap reject).
2. Bytes are streamed to disk with a hard cap, aborting mid-transfer.
3. The magic bytes of what landed on disk are checked against the extension.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)

#: Extension -> (human label, accepted MIME types)
SUPPORTED_TYPES: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    ".pdf": ("PDF document", ("application/pdf",)),
    ".docx": (
        "Word document",
        (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/zip",  # DOCX is a zip container
        ),
    ),
    ".txt": ("Plain text", ("text/plain",)),
    ".md": ("Markdown", ("text/plain", "text/markdown")),
    ".csv": ("CSV", ("text/plain", "text/csv")),
    ".png": ("PNG image", ("image/png",)),
    ".jpg": ("JPEG image", ("image/jpeg",)),
    ".jpeg": ("JPEG image", ("image/jpeg",)),
    ".tiff": ("TIFF image", ("image/tiff",)),
    ".tif": ("TIFF image", ("image/tiff",)),
}

ALLOWED_EXTENSIONS: Set[str] = set(SUPPORTED_TYPES)

#: Leading byte signatures, checked when python-magic is unavailable.
MAGIC_SIGNATURES: Tuple[Tuple[bytes, Tuple[str, ...]], ...] = (
    (b"%PDF-", (".pdf",)),
    (b"PK\x03\x04", (".docx",)),
    (b"\x89PNG\r\n\x1a\n", (".png",)),
    (b"\xff\xd8\xff", (".jpg", ".jpeg")),
    (b"II*\x00", (".tiff", ".tif")),
    (b"MM\x00*", (".tiff", ".tif")),
)

#: Extensions whose contents are free-form text and have no signature.
TEXT_EXTENSIONS = {".txt", ".md", ".csv"}

# python-magic is optional: it needs libmagic present on the host. When it is
# missing we fall back to the signature table above rather than skipping the
# check entirely.
try:  # pragma: no cover - depends on host libraries
    import magic as _magic

    _MAGIC_AVAILABLE = True
except Exception:  # pragma: no cover
    _magic = None
    _MAGIC_AVAILABLE = False


class FileValidationError(ValueError):
    """Raised when an upload is rejected. Message is safe to return to callers."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class ValidatedUpload:
    """Result of accepting an upload onto disk."""

    path: str
    size_bytes: int
    checksum: str
    extension: str
    detected_type: str


def normalise_extension(filename: str) -> str:
    """Return the lower-cased extension of ``filename``."""
    return os.path.splitext(filename or "")[1].lower()


def validate_filename(filename: Optional[str]) -> str:
    """Validate the filename and return its extension.

    Raises:
        FileValidationError: when the name is empty or the type unsupported.
    """
    if not filename or not filename.strip():
        raise FileValidationError("No filename provided.")

    # Reject path separators outright. Storage names are generated from a UUID
    # so traversal is not reachable today, but validating here keeps that true
    # if a future change ever echoes the original name into a path.
    if any(sep in filename for sep in ("/", "\\", "\x00")):
        raise FileValidationError("Filename must not contain path separators.")

    extension = normalise_extension(filename)
    if extension not in ALLOWED_EXTENSIONS:
        supported = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise FileValidationError(
            f"Unsupported file type '{extension or '(none)'}'. Supported: {supported}"
        )
    return extension


#: libmagic answers that carry no information. Treating these as a verdict
#: made uploads host-dependent: libmagic is present on Linux and usually absent
#: on Windows, so a file whose signature is recognisable but whose body is
#: truncated was accepted on one host and rejected with 400 on the other.
#: Fall through to the signature table instead of trusting a shrug.
INCONCLUSIVE_TYPES = frozenset(
    {"application/octet-stream", "application/x-empty", "inode/x-empty", "binary"}
)


def detect_content_type(head: bytes, extension: str) -> str:
    """Best-effort MIME detection from the first bytes of a file.

    libmagic wins when it identifies something specific — that is what catches
    a renamed executable. When it cannot tell, the signature table decides, so
    the same bytes are classified the same way on every host.
    """
    if _MAGIC_AVAILABLE:  # pragma: no cover - host dependent
        try:
            detected = _magic.from_buffer(head, mime=True)
            if detected and detected not in INCONCLUSIVE_TYPES:
                return detected
        except Exception as exc:
            logger.debug("libmagic detection failed, using signatures: %s", exc)

    for signature, extensions in MAGIC_SIGNATURES:
        if head.startswith(signature):
            return SUPPORTED_TYPES[extensions[0]][1][0]

    if extension in TEXT_EXTENSIONS and _looks_like_text(head):
        return "text/plain"
    return "application/octet-stream"


def _looks_like_text(head: bytes) -> bool:
    """True when the bytes decode as UTF-8 and hold no NUL characters."""
    if b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
        return True
    except UnicodeDecodeError:
        # A chunk boundary can split a multi-byte codepoint; tolerate the tail.
        try:
            head[:-4].decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False


def verify_content_matches_extension(detected: str, extension: str) -> None:
    """Reject files whose real content disagrees with their extension.

    Raises:
        FileValidationError: when the content type is not valid for ``extension``.
    """
    label, accepted = SUPPORTED_TYPES[extension]

    if extension in TEXT_EXTENSIONS:
        if detected.startswith("text/") or detected in accepted:
            return
        raise FileValidationError(
            f"File claims to be {label} but its contents are binary ({detected})."
        )

    if detected in accepted:
        return

    raise FileValidationError(
        f"File extension '{extension}' does not match its actual content "
        f"({detected}). Rename the file or upload the correct format."
    )


async def save_upload_stream(
    upload,
    destination: str,
    max_bytes: int,
    chunk_size: int = 1024 * 1024,
) -> ValidatedUpload:
    """Stream an ``UploadFile`` to ``destination``, validating as it goes.

    The size cap is enforced *during* the transfer, so an oversized upload is
    abandoned after ``max_bytes`` rather than being buffered in full. The
    partial file is removed before raising.

    Args:
        upload: A Starlette ``UploadFile``.
        destination: Absolute path to write to.
        max_bytes: Hard limit on accepted bytes.
        chunk_size: Read granularity.

    Returns:
        ValidatedUpload with the path, size, SHA-256 and detected type.

    Raises:
        FileValidationError: on oversize, empty, or mistyped content.
    """
    import aiofiles

    extension = validate_filename(upload.filename)
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    digest = hashlib.sha256()
    total = 0
    head = b""

    try:
        async with aiofiles.open(destination, "wb") as sink:
            while True:
                chunk = await upload.read(chunk_size)
                if not chunk:
                    break

                total += len(chunk)
                if total > max_bytes:
                    raise FileValidationError(
                        f"File exceeds the maximum size of "
                        f"{max_bytes // (1024 * 1024)}MB.",
                        status_code=413,
                    )

                if len(head) < 4096:
                    head += chunk[: 4096 - len(head)]

                digest.update(chunk)
                await sink.write(chunk)

        if total == 0:
            raise FileValidationError("Uploaded file is empty.")

        detected = detect_content_type(head, extension)
        verify_content_matches_extension(detected, extension)

    except Exception:
        _remove_quietly(destination)
        raise

    return ValidatedUpload(
        path=destination,
        size_bytes=total,
        checksum=digest.hexdigest(),
        extension=extension,
        detected_type=detected,
    )


def _remove_quietly(path: str) -> None:
    """Delete ``path``, ignoring the case where it was never created."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as exc:  # pragma: no cover - platform specific
        logger.warning("Could not remove partial upload %s: %s", path, exc)
