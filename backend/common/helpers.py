"""
Document parsing helpers for Roneira AI.
Moved from document_processor.py to enforce separation of concerns.
"""

import os
import re
import aiofiles
import logging
from typing import Dict, Any, Tuple
from datetime import datetime

# Optional imports for PDF/DOCX
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

logger = logging.getLogger(__name__)

async def extract_text_from_file(file_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Extract text from various file formats (PDF, DOCX, TXT).
    Returns tuple of (extracted_text, metadata).
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)

    metadata = {
        "filename": filename,
        "file_type": file_extension,
        "file_size": os.path.getsize(file_path),
        "processing_time": datetime.utcnow().isoformat(),
    }

    try:
        if file_extension == ".pdf":
            text, pdf_metadata = await _extract_from_pdf(file_path)
            metadata.update(pdf_metadata)
            return text, metadata

        elif file_extension in [".doc", ".docx"]:
            text, doc_metadata = await _extract_from_docx(file_path)
            metadata.update(doc_metadata)
            return text, metadata

        elif file_extension == ".txt":
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                text = await f.read()
            metadata.update({"pages": 1, "word_count": len(text.split())})
            return text, metadata

        elif file_extension in [".png", ".jpg", ".jpeg"]:
            metadata.update({"type": "image", "ocr_available": False})
            return (
                "Image file uploaded - OCR not implemented in this helper yet",
                metadata,
            )
        else:
            return f"Unsupported file type: {file_extension}", metadata

    except Exception as e:
        logger.error(f"Text extraction failed for {file_path}: {e}", exc_info=True)
        return f"Error extracting text: {str(e)}", metadata


async def _extract_from_pdf(file_path: str) -> Tuple[str, Dict[str, Any]]:
    """Extract text from PDF file using pdfplumber or PyPDF2 fallback."""
    text = ""
    metadata = {"pages": 0, "word_count": 0}

    try:
        if pdfplumber:
            with pdfplumber.open(file_path) as pdf:
                metadata["pages"] = len(pdf.pages)
                page_texts = []
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text() or ""
                        page_texts.append(f"--- Page {page_num + 1} ---\n{page_text}")
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                        page_texts.append(f"--- Page {page_num + 1} ---\n[Text extraction failed]")
                text = "\n".join(page_texts)
        elif PyPDF2:
            # Fallback to PyPDF2
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata["pages"] = len(pdf_reader.pages)
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text() or ""
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                        text += f"\n--- Page {page_num + 1} ---\n[Text extraction failed]\n"
        else:
            return "No PDF library installed (pdfplumber or PyPDF2)", metadata

        # Clean up text
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        metadata["word_count"] = len(text.split())
        return text, metadata

    except Exception as e:
        logger.error(f"PDF extraction failed: {e}", exc_info=True)
        return f"PDF extraction error: {str(e)}", metadata


async def _extract_from_docx(file_path: str) -> Tuple[str, Dict[str, Any]]:
    """Extract text from DOCX file."""
    if not DocxDocument:
        return "python-docx library not installed", {"pages": 0}

    try:
        doc = DocxDocument(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"

        metadata = {
            "pages": 1,
            "paragraphs": len(doc.paragraphs),
            "word_count": len(text.split()),
        }
        return text.strip(), metadata

    except Exception as e:
        logger.error(f"DOCX extraction failed: {e}", exc_info=True)
        return f"DOCX extraction error: {str(e)}", {"pages": 0, "word_count": 0}
