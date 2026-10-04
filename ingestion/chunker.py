"""
Chunker module for creating overlapping text chunks from page-level document data.
Preserves page boundaries, study IDs, source filenames, and precise metadata.
"""

from typing import List, Dict, Any


def chunk_document(
    doc_data: Dict[str, Any],
    study_id: str,
    target_words: int = 1000,
    overlap_words: int = 125,
    min_words: int = 200,
) -> List[Dict[str, Any]]:
    """
    Chunk a document's extracted pages into word-bounded overlapping chunks.

    Args:
        doc_data: Dictionary returned by load_pdf containing source_file and pages.
        study_id: Document study identifier (e.g., EMP-001).
        target_words: Target target size in words per chunk (default 1000, range ~800-1200).
        overlap_words: Target word overlap between consecutive chunks (default 125).
        min_words: Minimum words threshold to avoid trailing micro-chunks.

    Returns:
        List of chunk dictionaries with metadata:
            chunk_id, study_id, source_file, page_start, page_end, text
    """
    source_file = doc_data["source_file"]
    pages = doc_data["pages"]

    # 1. Linearize words and paragraph tokens with page tracking
    # Each token unit is a tuple: (word_str, page_number)
    token_units = []
    for page_info in pages:
        p_num = page_info["page_number"]
        text = page_info["text"]
        if not text.strip():
            continue

        # Split into words keeping page tracking
        words = text.split()
        for w in words:
            token_units.append((w, p_num))

    if not token_units:
        return []

    chunks = []
    total_tokens = len(token_units)
    start_idx = 0
    chunk_index = 1

    step_size = max(1, target_words - overlap_words)

    while start_idx < total_tokens:
        end_idx = min(start_idx + target_words, total_tokens)

        # If remaining tokens are small and we already have chunks, extend the previous chunk or stop
        remaining = total_tokens - start_idx
        if remaining < min_words and chunks:
            # Append remaining tokens to the last chunk if remaining words are very few
            last_chunk = chunks[-1]
            extra_words = [t[0] for t in token_units[start_idx:total_tokens]]
            extra_pages = [t[1] for t in token_units[start_idx:total_tokens]]
            last_chunk["text"] += " " + " ".join(extra_words)
            last_chunk["page_end"] = max(last_chunk["page_end"], max(extra_pages))
            break

        chunk_units = token_units[start_idx:end_idx]
        chunk_text = " ".join([unit[0] for unit in chunk_units])
        page_start = chunk_units[0][1]
        page_end = chunk_units[-1][1]

        chunk_id = f"{study_id}_CHK_{chunk_index:03d}"

        chunks.append({
            "chunk_id": chunk_id,
            "study_id": study_id,
            "source_file": source_file,
            "page_start": page_start,
            "page_end": page_end,
            "text": chunk_text
        })

        chunk_index += 1

        if end_idx == total_tokens:
            break

        start_idx += step_size

    return chunks
