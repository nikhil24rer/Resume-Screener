"""Robust PDF text extraction with graceful failure.

Primary parser: pdfplumber.  Fallback: PyMuPDF (if installed).
Scanned (image-only) PDFs are detected and rejected with a clear message;
OCR is intentionally NOT implemented (documented limitation).
"""
from __future__ import annotations

import io
from typing import BinaryIO, Union

import pdfplumber

MAX_FILE_MB = 10
MIN_TEXT_CHARS = 80  # below this we assume the PDF has no usable text layer


class ResumeParseError(Exception):
    """Raised with a user-friendly message when a file cannot be used."""


def _read_bytes(source: Union[bytes, BinaryIO]) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    data = source.read()
    try:
        source.seek(0)
    except Exception:
        pass
    return data


def _with_pdfplumber(data: bytes) -> str:
    pages = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        if len(pdf.pages) == 0:
            raise ResumeParseError("The PDF has no pages.")
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _with_pymupdf(data: bytes) -> str:
    import fitz  # PyMuPDF; imported lazily because it is optional

    with fitz.open(stream=data, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


def extract_text_from_pdf(source: Union[bytes, BinaryIO]) -> str:
    """Return the text of a PDF or raise ResumeParseError with a readable reason."""
    data = _read_bytes(source)

    if not data:
        raise ResumeParseError("The file is empty.")
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise ResumeParseError(f"File is larger than {MAX_FILE_MB} MB.")
    if not data.lstrip()[:5] == b"%PDF-":
        raise ResumeParseError("Not a valid PDF (missing PDF header). Is it a renamed .docx/.txt/.jpg?")

    text, errors = "", []
    for parser in (_with_pdfplumber, _with_pymupdf):
        try:
            text = parser(data)
            if len(text.strip()) >= MIN_TEXT_CHARS:
                break
        except ResumeParseError:
            raise
        except ImportError:
            continue  # optional fallback not installed
        except Exception as exc:  # corrupted / encrypted / malformed
            errors.append(f"{parser.__name__}: {exc}")

    if len(text.strip()) < MIN_TEXT_CHARS:
        if errors:
            raise ResumeParseError("The PDF could not be read (corrupted or password-protected).")
        raise ResumeParseError(
            "No extractable text found - this looks like a scanned/image-only PDF. "
            "OCR is not supported in this prototype."
        )
    return text
