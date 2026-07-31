"""OCR for scanned pages and image uploads.

Scope is deliberately narrow: rasterise a page, read the text off it, report
honestly when that is not possible. Everything else — preprocessing pipelines,
multiple competing engines, confidence voting — is absent until there is
evidence it improves an answer.

Dependencies are the ones already in the tree. ``pypdfium2`` arrives with
pdfplumber and renders a page without an external toolchain; ``Pillow`` is
already a requirement. Only ``pytesseract`` is new, and it is a thin wrapper:
the actual OCR is done by the **tesseract binary**, which is a system package
and not installable with pip. When that binary is missing, OCR reports itself
unavailable and the caller degrades visibly — the same contract the embedding
and LLM services follow.

``backend/services/free_ocr_service.py`` covers similar ground but imports
opencv, PyMuPDF and EasyOCR at module scope, none of which are dependencies of
this project. It stays unwired; see docs/ARCHITECTURE.md.

This lives in ``common`` rather than ``services`` on purpose: it is a detail of
text extraction, used only by ``common/helpers.py``. Importing it from
``backend.services`` would load that package's ``__init__``, which imports
``document_processor``, which imports ``helpers`` — a cycle.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from backend.core.config import get_settings

try:
    import pytesseract
except ImportError:  # pragma: no cover - optional dependency
    pytesseract = None

try:
    import pypdfium2
except ImportError:  # pragma: no cover - optional dependency
    pypdfium2 = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None

logger = logging.getLogger(__name__)

#: A page with fewer than this many non-whitespace characters is treated as
#: having no usable text layer, and is a candidate for OCR. Scanned pages
#: routinely yield a handful of stray characters rather than a clean zero.
MIN_CHARS_FOR_TEXT_LAYER = 24

#: Rendering scale. Tesseract wants roughly 300 DPI; PDF user space is 72 DPI.
_RENDER_SCALE = 300 / 72

#: Cap on pages rasterised for one document. Rendering and OCR are the most
#: expensive things this service does, and an unbounded loop over a 900-page
#: scan would occupy a worker indefinitely.
MAX_OCR_PAGES = 50


@lru_cache(maxsize=1)
def ocr_availability() -> Tuple[bool, str]:
    """Return ``(available, reason)`` for OCR.

    Cached: probing the binary spawns a subprocess, and the answer cannot
    change without a restart. The reason is written to be shown to a user.
    """
    settings = get_settings()

    if not settings.enable_ocr:
        return False, "OCR is disabled (ENABLE_OCR=false)."
    if pytesseract is None:
        return False, "pytesseract is not installed on the server."
    if Image is None:
        return False, "Pillow is not installed on the server."

    # An explicit path wins; otherwise rely on PATH.
    configured = (settings.tesseract_path or "").strip()
    if configured:
        if not os.path.isfile(configured):
            return False, f"TESSERACT_PATH points at nothing: {configured}"
        pytesseract.pytesseract.tesseract_cmd = configured

    try:
        version = pytesseract.get_tesseract_version()
    except Exception:
        # Covers TesseractNotFoundError and anything the wrapper raises when
        # the binary is present but unusable.
        return False, (
            "The tesseract binary was not found. It is a system package, not a "
            "pip install — see the OCR section of the README."
        )

    return True, f"tesseract {version}"


def ocr_is_available() -> bool:
    return ocr_availability()[0]


def _image_to_text(image) -> str:
    """Run tesseract over one PIL image."""
    return pytesseract.image_to_string(image) or ""


def _ocr_image_bytes_sync(data: bytes) -> str:
    with Image.open(io.BytesIO(data)) as image:
        # Tesseract is markedly better on greyscale than on a colour scan.
        return _image_to_text(image.convert("L"))


def _ocr_pdf_pages_sync(file_path: str, page_numbers: List[int]) -> Dict[int, str]:
    """Rasterise the given 1-based pages and OCR each one."""
    results: Dict[int, str] = {}
    pdf = pypdfium2.PdfDocument(file_path)
    try:
        for page_number in page_numbers:
            page = pdf[page_number - 1]
            try:
                bitmap = page.render(scale=_RENDER_SCALE, grayscale=True)
                image = bitmap.to_pil()
                try:
                    text = _image_to_text(image)
                finally:
                    image.close()
                if text.strip():
                    results[page_number] = text
            except Exception as exc:
                # One bad page must not lose the rest of the document.
                logger.warning("OCR failed on page %s: %s", page_number, exc)
            finally:
                page.close()
    finally:
        pdf.close()
    return results


async def ocr_pdf_pages(file_path: str, page_numbers: List[int]) -> Dict[int, str]:
    """OCR selected pages of a PDF.

    Returns a ``{page_number: text}`` map containing only pages that produced
    text. Missing pages are not an error — a genuinely blank scan yields
    nothing, and that is a real answer.
    """
    if not page_numbers or not ocr_is_available() or pypdfium2 is None:
        return {}

    capped = page_numbers[:MAX_OCR_PAGES]
    if len(page_numbers) > MAX_OCR_PAGES:
        logger.warning(
            "Document has %s pages needing OCR; processing the first %s.",
            len(page_numbers),
            MAX_OCR_PAGES,
        )

    # tesseract and page rendering are blocking and CPU-bound; keep them off
    # the event loop so the API stays responsive during a large upload.
    return await asyncio.to_thread(_ocr_pdf_pages_sync, file_path, capped)


async def ocr_image_file(file_path: str) -> Optional[str]:
    """OCR a standalone image upload. ``None`` when OCR is unavailable."""
    if not ocr_is_available():
        return None

    def _run() -> str:
        with open(file_path, "rb") as handle:
            return _ocr_image_bytes_sync(handle.read())

    try:
        return await asyncio.to_thread(_run)
    except Exception as exc:
        logger.warning("OCR failed for image %s: %s", file_path, exc)
        return None
