"""
Process Documents Pipeline Script
Extracts, cleans, saves text, chunks documents, saves JSONL & metadata,
and performs pipeline validation checks.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

from document_loader import load_pdf
from chunker import chunk_document


# Corpus Mapping
CORPUS_FILES = [
    ("EMP-001", "EMP-001.pdf"),
    ("EMP-002", "EMP-002.pdf"),
    ("EMP-003", "EMP-003.pdf"),
    ("EMP-004", "EMP-004.pdf"),
    ("EMP-005", "EMP-005.pdf"),
]

PDF_DIR = Path("data/raw/pdf")
TEXT_OUT_DIR = Path("data/processed/text")
CHUNKS_OUT_DIR = Path("data/processed/chunks")


def validate_chunks(chunks: List[Dict[str, Any]]) -> None:
    """Validate chunk integrity against core requirements."""
    seen_ids = set()
    for idx, c in enumerate(chunks):
        c_id = c.get("chunk_id")
        if not c_id:
            raise ValueError(f"Chunk at index {idx} missing 'chunk_id'")
        if c_id in seen_ids:
            raise ValueError(f"Duplicate chunk_id found: {c_id}")
        seen_ids.add(c_id)

        if not c.get("study_id"):
            raise ValueError(f"Chunk {c_id} missing 'study_id'")
        if not c.get("source_file"):
            raise ValueError(f"Chunk {c_id} missing 'source_file'")
        if c.get("page_start") is None or c.get("page_end") is None:
            raise ValueError(f"Chunk {c_id} missing page information")
        if not c.get("text") or not c["text"].strip():
            raise ValueError(f"Chunk {c_id} has empty text")


def main():
    TEXT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    all_chunks = []
    document_summaries = []

    print("=" * 80)
    print("STARTING DOCUMENT PROCESSING PIPELINE")
    print("=" * 80)

    for study_id, filename in CORPUS_FILES:
        pdf_path = PDF_DIR / filename
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing required PDF: {pdf_path}")

        print(f"Processing {study_id} ({filename})...")
        doc_data = load_pdf(str(pdf_path))

        # Build clean page-level document text
        page_texts = []
        total_chars = 0
        for p in doc_data["pages"]:
            p_num = p["page_number"]
            p_text = p["text"]
            total_chars += len(p_text)
            page_texts.append(f"--- Page {p_num} ---\n{p_text}")

        full_doc_text = "\n\n".join(page_texts)

        # 1. Save cleaned text file
        text_out_file = TEXT_OUT_DIR / f"{study_id}.txt"
        with open(text_out_file, "w", encoding="utf-8") as f:
            f.write(full_doc_text)

        # 2. Chunk document
        chunks = chunk_document(doc_data, study_id=study_id)
        all_chunks.extend(chunks)

        pages_count = doc_data["total_pages"]
        chunks_count = len(chunks)

        summary_rows.append({
            "study_id": study_id,
            "filename": filename,
            "pages": pages_count,
            "chars": total_chars,
            "chunks": chunks_count
        })

        document_summaries.append({
            "study_id": study_id,
            "source_file": filename,
            "total_pages": pages_count,
            "total_characters": total_chars,
            "chunk_count": chunks_count,
            "chunk_ids": [c["chunk_id"] for c in chunks]
        })

    # 3. Validate all chunks
    validate_chunks(all_chunks)
    print("\n[OK] Validation passed: No empty chunks, no duplicate IDs, all metadata intact.")

    # 4. Save JSONL chunks file
    jsonl_out = CHUNKS_OUT_DIR / "chunks.jsonl"
    with open(jsonl_out, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Saved {len(all_chunks)} chunks to {jsonl_out}")

    # 5. Save chunk metadata summary JSON file
    metadata_out = CHUNKS_OUT_DIR / "chunk_metadata.json"
    chunk_meta = {
        "total_documents": len(CORPUS_FILES),
        "total_chunks": len(all_chunks),
        "chunking_strategy": {
            "target_words": 1000,
            "overlap_words": 125,
            "range": "800-1200 words"
        },
        "documents": document_summaries
    }

    with open(metadata_out, "w", encoding="utf-8") as f:
        json.dump(chunk_meta, f, indent=2, ensure_ascii=False)
    print(f"Saved chunk metadata summary to {metadata_out}")

    # 6. Print Summary Table
    print("\n" + "=" * 80)
    print(f"{'Document':<12} | {'Pages':<8} | {'Characters':<12} | {'Chunks':<8}")
    print("-" * 80)
    for r in summary_rows:
        print(f"{r['study_id']:<12} | {r['pages']:<8} | {r['chars']:<12} | {r['chunks']:<8}")
    print("=" * 80)
    print(f"TOTAL: {len(summary_rows)} documents | {sum(r['pages'] for r in summary_rows)} pages | {sum(r['chars'] for r in summary_rows)} chars | {sum(r['chunks'] for r in summary_rows)} chunks")
    print("=" * 80)


if __name__ == "__main__":
    main()
