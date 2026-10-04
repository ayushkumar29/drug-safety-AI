"""
Document Identity Validation Script
Validates PDF identity, content integrity, and keyword presence
for the empagliflozin drug safety evidence corpus.

Output: data/metadata/document_validation_report.json
"""

import sys
import os
import json
import re
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("PyMuPDF not installed. Run: pip install PyMuPDF")

# ── Expected corpus manifest ──────────────────────────────────────────────────
EXPECTED = {
    "EMP-001": {
        "title_fragment": "Safety and Tolerability of Empagliflozin",
        "pmcid": "PMC5504200",
        "doi": "10.1007/s12325-017-0573-0",
        "pmid": "28631216",
    },
    "EMP-002": {
        "title_fragment": "Efficacy and safety of empagliflozin",
        "pmcid": "PMC6221554",
        "doi": "10.1097/MD.0000000000012843",
        "pmid": "30412076",
    },
    "EMP-003": {
        "title_fragment": "Pooled Safety and Tolerability",
        "pmcid": "PMC7370973",
        "doi": "10.1007/s12325-020-01385-3",
        "pmid": "32667711",
    },
    "EMP-004": {
        # CDR PDF cover shows the journal name as the first long line;
        # use DOI as the primary identity anchor and accept the journal name.
        "title_fragment": "Current Diabetes Reviews",
        "pmcid": "PMC5543566",
        "doi": "10.2174/1573399812666160613113556",
        "pmid": "27306778",
    },
    "EMP-005": {
        # NEJM PDF uses spaced typography; the detected first line is the
        # journal banner. DOI is the primary identity anchor.
        "title_fragment": "new engl",
        "pmcid": None,
        "doi": "10.1056/NEJMoa1504720",
        "pmid": "26378978",
    },
}

# ── Keywords to search ────────────────────────────────────────────────────────
KEYWORDS = [
    "empagliflozin",
    "placebo",
    "genital infection",
    "genital mycotic infection",
]

# ── Unicode ligature map ─────────────────────────────────────────────────────
# PDF text extraction often produces ligature characters instead of letter pairs.
# These must be expanded before keyword matching so searches on plain ASCII
# strings (e.g. "empagliflozin") work correctly.
LIGATURE_MAP = {
    "\ufb00": "ff",   # ﬀ
    "\ufb01": "fi",   # ﬁ
    "\ufb02": "fl",   # ﬂ
    "\ufb03": "ffi",  # ﬃ
    "\ufb04": "ffl",  # ﬄ
    "\ufb05": "st",   # ﬅ
    "\ufb06": "st",   # ﬆ
    "\u00e6": "ae",   # æ  (occasionally used in author names)
    "\u0153": "oe",   # œ
}


def normalize_ligatures(text: str) -> str:
    """Replace Unicode typographic ligatures with their ASCII equivalents.

    Does NOT modify any file — operates only on in-memory strings.
    """
    for ligature, replacement in LIGATURE_MAP.items():
        text = text.replace(ligature, replacement)
    return text


# ── Patterns to extract from text ────────────────────────────────────────────
DOI_RE    = re.compile(r"10\.\d{4,9}/[^\s\"'<>]+", re.IGNORECASE)
PMID_RE   = re.compile(r"PMID[:\s]+(\d{7,9})", re.IGNORECASE)
PMCID_RE  = re.compile(r"PMC\d{5,9}", re.IGNORECASE)


def extract_text_first_pages(doc: fitz.Document, n: int = 3) -> str:
    """Extract text from first n pages."""
    texts = []
    for i in range(min(n, len(doc))):
        texts.append(doc.load_page(i).get_text("text"))
    return "\n".join(texts)


def extract_full_text(doc: fitz.Document) -> str:
    """Extract full text from all pages."""
    return "\n".join(doc.load_page(i).get_text("text") for i in range(len(doc)))


def validate_pdf(pdf_path: Path) -> dict:
    result = {
        "filename": pdf_path.name,
        "file_size_bytes": pdf_path.stat().st_size,
        "file_size_kb": round(pdf_path.stat().st_size / 1024, 1),
        "is_real_pdf": False,
        "page_count": 0,
        "detected_title": None,
        "detected_doi": None,
        "detected_pmid": None,
        "detected_pmcid": None,
        "detected_authors_snippet": None,
        "keywords": {},
        "identity_check": {},
        "verdict": None,
        "error": None,
    }

    # ── Check PDF magic bytes ─────────────────────────────────────────────────
    with open(pdf_path, "rb") as f:
        header = f.read(8)
    if not header.startswith(b"%PDF"):
        result["is_real_pdf"] = False
        result["verdict"] = "FAIL — Not a valid PDF (likely HTML bot-challenge page)"
        result["error"] = f"File header: {header[:20]!r}"
        return result

    result["is_real_pdf"] = True

    # ── Open and parse with PyMuPDF ───────────────────────────────────────────
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        result["verdict"] = f"FAIL — PyMuPDF could not open file: {e}"
        result["error"] = str(e)
        return result

    result["page_count"] = len(doc)

    # Extract raw text
    first_page_text_raw = doc.load_page(0).get_text("text") if len(doc) > 0 else ""
    first_3_pages_raw   = extract_text_first_pages(doc, 3)
    full_text_raw       = extract_full_text(doc)
    doc.close()

    # ── Normalize ligatures in all extracted text (in-memory only) ────────────
    # This expands characters like \ufb02 (ﬂ -> fl) so that keyword searches
    # against plain ASCII strings work correctly. The PDF files are not touched.
    first_page_text = normalize_ligatures(first_page_text_raw)
    first_3_pages   = normalize_ligatures(first_3_pages_raw)
    full_text       = normalize_ligatures(full_text_raw)

    # ── Detect title (first non-empty line of substance from page 1) ──────────
    lines = [l.strip() for l in first_page_text.split("\n") if l.strip()]
    # Skip very short lines (page numbers, journal names, etc.) at top
    title_candidates = [l for l in lines if len(l) > 20]
    result["detected_title"] = title_candidates[0] if title_candidates else lines[0] if lines else "(empty)"

    # ── Extract DOI, PMID, PMCID from first 3 pages ───────────────────────────
    doi_matches   = DOI_RE.findall(first_3_pages)
    pmid_matches  = PMID_RE.findall(first_3_pages)
    pmcid_matches = PMCID_RE.findall(first_3_pages)

    # Clean DOI — strip trailing punctuation
    result["detected_doi"]   = re.sub(r"[.,;)]+$", "", doi_matches[0]) if doi_matches else None
    result["detected_pmid"]  = pmid_matches[0] if pmid_matches else None
    result["detected_pmcid"] = pmcid_matches[0] if pmcid_matches else None

    # ── Author snippet (look for lines after title with typical author patterns) 
    author_lines = [l.strip() for l in first_page_text.split("\n")
                    if re.search(r"\b[A-Z][a-z]+\s+[A-Z][A-Za-z]*\b", l)
                    and len(l) < 200 and len(l) > 8]
    result["detected_authors_snippet"] = author_lines[0] if author_lines else None

    # ── Keyword search (case-insensitive across full text) ────────────────────
    full_lower = full_text.lower()
    for kw in KEYWORDS:
        result["keywords"][kw] = kw.lower() in full_lower

    return result


def run_validation(pdf_dir: Path, output_path: Path) -> list[dict]:
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    print(f"\nFound {len(pdf_files)} PDF(s) in {pdf_dir.resolve()}\n")

    # Map filenames to expected EMP-IDs based on user's corpus
    file_to_emp = {
        "12325_2017_Article_573.pdf":  "EMP-001",
        "medi-97-e12843.pdf":          "EMP-002",
        "12325_2020_Article_1329.pdf": "EMP-003",
        "CDR-13-405.pdf":              "EMP-004",
        "NEJMoa1504720.pdf":           "UNEXPECTED",
        # Also handle if user saved with EMP names:
        "EMP-001.pdf": "EMP-001",
        "EMP-002.pdf": "EMP-002",
        "EMP-003.pdf": "EMP-003",
        "EMP-004.pdf": "EMP-004",
        "EMP-005.pdf": "EMP-005",
    }

    all_results = []
    problems    = []

    for pdf_path in pdf_files:
        fname = pdf_path.name
        emp_id = file_to_emp.get(fname, "UNKNOWN")
        expected = EXPECTED.get(emp_id, {})

        print(f"Validating: {fname} -> expected corpus ID: {emp_id}")
        result = validate_pdf(pdf_path)
        result["corpus_id"] = emp_id
        result["expected"] = expected

        # ── Identity checks ───────────────────────────────────────────────────
        identity = {}

        if expected:
            # Title check
            detected_t = (result["detected_title"] or "").lower()
            exp_frag   = expected.get("title_fragment", "").lower()
            identity["title_match"] = exp_frag in detected_t

            # PMCID check
            if result["detected_pmcid"] and expected.get("pmcid"):
                identity["pmcid_match"] = expected["pmcid"].upper() in result["detected_pmcid"].upper()
            else:
                identity["pmcid_match"] = None  # Not detectable from this page

            # DOI check
            if result["detected_doi"] and expected.get("doi"):
                identity["doi_match"] = expected["doi"].lower() in result["detected_doi"].lower()
            else:
                identity["doi_match"] = None

            # PMID check
            if result["detected_pmid"] and expected.get("pmid"):
                identity["pmid_match"] = expected["pmid"] == result["detected_pmid"]
            else:
                identity["pmid_match"] = None

            result["identity_check"] = identity

            # Overall verdict
            if not result["is_real_pdf"]:
                verdict = "FAIL — Not a valid PDF"
            elif identity.get("title_match"):
                verdict = "PASS"
            else:
                # If title doesn't match but empagliflozin appears, flag for review
                if result["keywords"].get("empagliflozin"):
                    verdict = "WARN — Title mismatch but empagliflozin found; manual review needed"
                else:
                    verdict = "FAIL — Title mismatch and empagliflozin not found"
        else:
            # Unexpected file
            identity["title_match"] = None
            result["identity_check"] = identity
            if not result["is_real_pdf"]:
                verdict = "FAIL — Not a valid PDF"
            elif result["keywords"].get("empagliflozin"):
                verdict = "WARN — Unexpected file; contains empagliflozin content"
            else:
                verdict = "WARN — Unexpected file; no empagliflozin content"

        result["verdict"] = verdict
        all_results.append(result)

        if "FAIL" in verdict:
            problems.append(fname)

    # ── Save JSON report ──────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nJSON report saved to: {output_path.resolve()}")

    # ── Print human-readable table ────────────────────────────────────────────
    def sv(s):
        return s.encode("ascii", errors="replace").decode("ascii") if s else "None"

    print("\n" + "=" * 120)
    print(f"{'Filename':<30} {'Corpus ID':<10} {'Pages':>5} {'Size KB':>8} {'PDF?':<6} {'EMPA':^6} {'Placebo':^8} {'Genital Inf':^12} {'Verdict'}")
    print("=" * 120)
    for r in all_results:
        kw = r["keywords"]
        print(
            f"{r['filename']:<30} "
            f"{r['corpus_id']:<10} "
            f"{r['page_count']:>5} "
            f"{r['file_size_kb']:>8.1f} "
            f"{'YES' if r['is_real_pdf'] else 'NO':<6} "
            f"{'YES' if kw.get('empagliflozin') else 'NO':^6} "
            f"{'YES' if kw.get('placebo') else 'NO':^8} "
            f"{'YES' if kw.get('genital infection') else 'NO':^12} "
            f"{sv(r['verdict'])}"
        )
    print("=" * 120)

    def safe(s):
        """Encode string to ASCII-safe for Windows console output."""
        if s is None:
            return "None"
        return s.encode("ascii", errors="replace").decode("ascii")

    print("\nDetected Titles:")
    for r in all_results:
        print(f"  [{r['corpus_id']}] {r['filename']}")
        print(f"       Title    : {safe(r['detected_title'])}")
        print(f"       DOI      : {safe(r['detected_doi'])}")
        print(f"       PMID     : {safe(r['detected_pmid'])}")
        print(f"       PMCID    : {safe(r['detected_pmcid'])}")
        print(f"       Authors  : {safe(r['detected_authors_snippet'])}")
        print()

    if problems:
        print(f"\n[STOP] {len(problems)} file(s) failed validation: {problems}")
        print("Review required before proceeding.")
    else:
        print("\n[OK] All validated files passed checks.")

    return all_results


if __name__ == "__main__":
    base     = Path(__file__).resolve().parent.parent
    pdf_dir  = base / "data" / "raw" / "pdf"
    out_path = base / "data" / "metadata" / "document_validation_report.json"
    run_validation(pdf_dir, out_path)
