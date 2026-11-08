"""PDF text extraction and line-cleaning utilities for Indonesian regulations."""

from __future__ import annotations

import re
from pathlib import Path

import pypdfium2 as pdfium


_PAGE_HEADER_RE = re.compile(
    r"^\s*-\s*\d+\s*-\s*$|^\s*\d+\s*$"
    r"|^\s*PRESIDEN\s*$|^\s*REPUBLIK\s+INDONESIA\s*$"
    r"|^\s*MENTERI\b.*$|^\s*SALINAN\s*$"
    r"|^\s*www\.\S+\s*$|^\s*jdih\.\S+\s*$",
    re.IGNORECASE,
)


def extract_pdf_pages(pdf_path: Path) -> list[str]:
    """Extract raw text per page from a PDF file using pypdfium2."""
    doc = pdfium.PdfDocument(str(pdf_path))
    pages: list[str] = []
    for idx in range(len(doc)):
        page = doc[idx]
        textpage = page.get_textpage()
        pages.append(textpage.get_text_bounded())
        textpage.close()
        page.close()
    doc.close()
    return pages


def clean_page_text(raw: str) -> str:
    """Strip recurring page headers, footers, and watermarks from a single page."""
    out_lines: list[str] = []
    for line in raw.splitlines():
        line_clean = line.replace("\r", "").strip()
        if _PAGE_HEADER_RE.match(line_clean):
            continue
        out_lines.append(line.replace("\r", ""))
    return "\n".join(out_lines)


def extract_clean_text(pdf_path: Path) -> tuple[str, int]:
    """Return ``(cleaned_full_text, page_count)`` for a regulatory PDF."""
    pages = extract_pdf_pages(pdf_path)
    cleaned = [clean_page_text(p) for p in pages]
    return "\n".join(cleaned), len(pages)
