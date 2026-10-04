import re
from typing import List, Dict, Any
from pathlib import Path

try:
    from rag.embeddings import EmbeddingManager
    from rag.vector_store import VectorStore, INDEX_DIR
except ImportError:
    from embeddings import EmbeddingManager
    from vector_store import VectorStore, INDEX_DIR



def compute_quantitative_score(text: str) -> tuple:
    """
    Computes a deterministic score (0.0 to 1.0) indicating presence of quantitative clinical evidence.
    Returns (score, evidence_type).
    """
    if not text:
        return 0.0, "none"

    # Raw event count & sample size indicators (highest priority)
    # Includes table format counts like 42 (1.8) or 153 (6.5) with optional % inside parentheses
    raw_event_pattern = re.compile(
        r'\b(?:\d{1,4}\s*\(\s*\d{1,2}(?:\.\d+)?\s*%?\s*\)|\d{1,4}\s*events\b|\b\d{1,4}\s*vs\.?\s*\d{1,4}\b|Rate/100|patient-years|n\s*=\s*\d+|N\s*=\s*\d+|\b\d{3,5}\s*patients\b)',
        re.IGNORECASE
    )
    # Summary & derived statistics indicators (secondary priority)
    derived_stat_pattern = re.compile(
        r'\b(?:RR|OR|HR|hazard ratio|risk ratio|odds ratio|95%\s*CI|confidence interval|p\s*<|p\s*=|p-value)\b',
        re.IGNORECASE
    )

    raw_matches = len(raw_event_pattern.findall(text))
    has_raw = raw_matches > 0
    has_derived = bool(derived_stat_pattern.search(text))

    if raw_matches >= 2:
        return 1.0, "raw_event_counts_table"
    elif has_raw and has_derived:
        return 1.0, "raw_event_counts_and_derived_stats"
    elif has_raw:
        return 0.85, "raw_event_counts"
    elif has_derived:
        return 0.4, "derived_statistics_only"
    elif "%" in text or re.search(r'\b\d+\b', text):
        return 0.2, "general_numbers"
    else:
        return 0.0, "none"


def compute_outcome_relevance_score(text: str, outcome: str = None) -> float:
    """
    Computes a deterministic score (0.0 to 1.0) measuring occurrence of target outcome in chunk text.
    Handles case, plurals/singulars, and token overlap.
    """
    if not text or not outcome or not outcome.strip():
        return 0.0

    text_clean = text.lower()
    outcome_clean = outcome.lower().strip()

    # Exact phrase match
    if outcome_clean in text_clean:
        return 1.0

    # Plural / singular simple stemming match (e.g., infection vs infections)
    if outcome_clean.rstrip('s') in text_clean:
        return 0.9

    # Token overlap match for multi-word outcomes (e.g. "genital infection" -> ["genital", "infection"])
    tokens = [t for t in re.split(r'\W+', outcome_clean) if len(t) > 2]
    if tokens:
        matches = sum(1 for t in tokens if t.rstrip('s') in text_clean)
        token_ratio = matches / len(tokens)
        if token_ratio == 1.0:
            return 0.85
        elif token_ratio >= 0.5:
            return 0.5

    return 0.0


def compute_table_cooccurrence_score(text: str, outcome: str = None) -> float:
    """
    Computes a deterministic score (0.0 to 1.0) measuring co-occurrence of target outcome
    and raw arm event counts/percentages on the same or adjacent lines in clinical tables.
    """
    if not text or not outcome or not outcome.strip():
        return 0.0

    text_clean = text.lower()
    outcome_clean = outcome.lower().strip()

    if outcome_clean not in text_clean and outcome_clean.rstrip('s') not in text_clean:
        return 0.0

    raw_pattern = re.compile(r'\b\d{1,4}\s*\(\s*\d{1,2}(?:\.\d+)?\s*%?\s*\)|\b\d{1,4}\s*events?\b', re.IGNORECASE)

    lines = text_clean.split('\n')
    for i, line in enumerate(lines):
        if outcome_clean in line or outcome_clean.rstrip('s') in line:
            context_block = " ".join(lines[max(0, i - 1):min(len(lines), i + 2)])
            if raw_pattern.search(context_block):
                return 1.0

    if raw_pattern.search(text_clean):
        return 0.50
    return 0.20


class Retriever:
    """
    Retriever class combining EmbeddingManager and VectorStore to execute similarity queries with outcome-aware quantitative reranking.
    """

    def __init__(self, vector_store: VectorStore = None, embedding_manager: EmbeddingManager = None):
        self.embedding_manager = embedding_manager or EmbeddingManager()
        if vector_store is None:
            self.vector_store = VectorStore.load(INDEX_DIR)
        else:
            self.vector_store = vector_store

    def retrieve(
        self, 
        query: str, 
        outcome: str = None, 
        top_k: int = 5, 
        rerank: bool = True, 
        fetch_multiplier: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Embed query string, retrieve candidate chunks, and optionally rerank using quantitative & outcome scores.

        Returns list of dicts with:
            chunk_id, study_id, source_file, page_start, page_end, similarity_score, quantitative_score, outcome_score, table_cooccurrence_score, combined_score, quantitative_evidence_type, text
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        # Multi-query strategy for robust semantic matching
        queries = [query]
        if outcome and outcome.strip():
            queries.append(f"{query} {outcome} quantitative event rate n (%) table")
            queries.append(f"{outcome} placebo events table")

        # Candidate-pool expansion: fetch up to total indexed vectors or top_k * fetch_multiplier
        fetch_k = max(top_k * fetch_multiplier, self.vector_store.total_vectors) if rerank else top_k

        chunk_sims = {}
        chunk_objs = {}
        for q_text in queries:
            q_vec = self.embedding_manager.embed_text(q_text)
            results = self.vector_store.search(q_vec, top_k=fetch_k)
            for res in results:
                cid = res["chunk_id"]
                sim = res["similarity_score"]
                if cid not in chunk_sims or sim > chunk_sims[cid]:
                    chunk_sims[cid] = sim
                    chunk_objs[cid] = res

        formatted_results = []
        for cid, res in chunk_objs.items():
            text = res["text"]
            sem_score = round(chunk_sims[cid], 4)
            quant_score, quant_type = compute_quantitative_score(text)
            out_score = compute_outcome_relevance_score(text, outcome=outcome)
            cooccur_score = compute_table_cooccurrence_score(text, outcome=outcome)
            
            # Combined score weighting:
            # When outcome is provided: 35% semantic similarity + 30% quantitative evidence + 20% outcome relevance + 15% table cooccurrence
            # Otherwise: 55% semantic similarity + 45% quantitative evidence
            if rerank:
                if outcome:
                    combined_score = round(0.35 * sem_score + 0.30 * quant_score + 0.20 * out_score + 0.15 * cooccur_score, 4)
                else:
                    combined_score = round(0.55 * sem_score + 0.45 * quant_score, 4)
            else:
                combined_score = sem_score

            formatted_results.append({
                "chunk_id": res["chunk_id"],
                "study_id": res["study_id"],
                "source_file": res["source_file"],
                "page_start": res["page_start"],
                "page_end": res["page_end"],
                "similarity_score": sem_score,
                "quantitative_score": quant_score,
                "outcome_score": round(out_score, 4),
                "table_cooccurrence_score": round(cooccur_score, 4),
                "combined_score": combined_score,
                "quantitative_evidence_type": quant_type,
                "text": text
            })

        if rerank:
            formatted_results.sort(key=lambda x: x["combined_score"], reverse=True)
            formatted_results = formatted_results[:top_k]

        return formatted_results


def retrieve(query: str, outcome: str = None, top_k: int = 25, rerank: bool = True) -> List[Dict[str, Any]]:
    """
    Convenience function to perform retrieval with default loaded VectorStore.
    Defaults to returning top 25 candidates for quantitative evidence extraction.
    """
    retriever = Retriever()
    return retriever.retrieve(query, outcome=outcome, top_k=top_k, rerank=rerank)




