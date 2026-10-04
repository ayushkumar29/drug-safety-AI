"""
Document Loader module for extracting and cleaning text from PDF files.
Preserves page boundaries, normalizes ligatures, cleans whitespace/headers/footers,
and preserves table/paragraph structure.
"""

import re
import pymupdf as fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any


# ── Unicode Ligature Map ──────────────────────────────────────────────────────
LIGATURE_MAP = {
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
    "\ufb00": "ff",
}


def normalize_ligatures(text: str) -> str:
    """Replace Unicode ligatures with their standard string equivalents."""
    for lig, replacement in LIGATURE_MAP.items():
        text = text.replace(lig, replacement)
    return text


def clean_page_text(text: str) -> str:
    """
    Clean raw page text:
    - Normalize Unicode ligatures
    - Replace non-breaking spaces
    - Standardize newline characters while preserving paragraph breaks
    - Normalize horizontal whitespace within lines without destroying layout
    """
    if not text:
        return ""

    # 1. Ligature normalization
    text = normalize_ligatures(text)

    # 2. Normalize non-breaking spaces and carriage returns
    text = text.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")

    # 3. Clean horizontal whitespace line-by-line while preserving empty lines (paragraphs)
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        # Strip trailing trailing whitespace, collapse multiple horizontal spaces
        line_clean = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(line_clean)

    # 4. Collapse 3 or more consecutive newlines down to 2 (paragraph boundary)
    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return cleaned_text.strip()


def remove_headers_footers(pages_text: List[str]) -> List[str]:
    """
    Identify and remove exact repeating header/footer lines occurring across pages.
    Preserves single-page content and unique text.
    """
    if len(pages_text) <= 2:
        return pages_text

    line_counts: Dict[str, int] = {}
    total_pages = len(pages_text)

    # Count occurrences of single-line header/footer candidates (short lines appearing at top/bottom of pages)
    for p_text in pages_text:
        lines = [l.strip() for l in p_text.split("\n") if l.strip()]
        if not lines:
            continue
        # Candidates: first 2 lines and last 2 lines
        candidates = set()
        if len(lines) >= 1:
            candidates.add(lines[0])
        if len(lines) >= 2:
            candidates.add(lines[1])
        if len(lines) >= 3:
            candidates.add(lines[-1])
        if len(lines) >= 4:
            candidates.add(lines[-2])

        for cand in candidates:
            # Only consider candidate line if it's typical header/footer length (< 120 chars) and not empty
            if 3 <= len(cand) <= 120:
                line_counts[cand] = line_counts.get(cand, 0) + 1

    # Lines occurring on >= 40% of pages are considered repeating headers/footers
    threshold = max(2, int(total_pages * 0.4))
    header_footer_patterns = {line for line, count in line_counts.items() if count >= threshold}

    processed_pages = []
    for p_text in pages_text:
        lines = p_text.split("\n")
        filtered_lines = []
        for l in lines:
            if l.strip() in header_footer_patterns:
                continue
            filtered_lines.append(l)
        processed_pages.append("\n".join(filtered_lines).strip())

    return processed_pages


def load_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Open PDF, extract page-by-page text, clean text, and return structured document data.

    Returns:
        Dict with keys:
            "source_file": filename
            "total_pages": int
            "pages": List[Dict] with "page_number" (1-indexed) and "text"
    """
    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        doc = fitz.open(str(pdf_path_obj))
    except Exception as e:
        raise ValueError(f"Failed to open PDF {pdf_path}: {e}")

    raw_pages = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        # Get extracted text preserving layout where possible
        page_text = page.get_text("text")
        cleaned_text = clean_page_text(page_text)
        raw_pages.append(cleaned_text)

    doc.close()

    # Remove repeating headers/footers
    cleaned_pages = remove_headers_footers(raw_pages)

    pages_data = []
    for idx, p_text in enumerate(cleaned_pages):
        pages_data.append({
            "page_number": idx + 1,
            "text": p_text
        })

    return {
        "source_file": pdf_path_obj.name,
        "total_pages": len(pages_data),
        "pages": pages_data
    }
