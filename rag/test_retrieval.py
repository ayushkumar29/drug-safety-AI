"""
Test Retrieval Module (Task 19A Regression & Audit Tests)
Builds vector index from data/processed/chunks/chunks.jsonl, runs required test queries A-F,
prints results and validates embedding/vector search pipeline requirements.
"""

import json
import unittest
import sys
from pathlib import Path
from typing import List, Dict, Any

try:
    from embeddings import EmbeddingManager
    from vector_store import VectorStore, INDEX_DIR
    from retriever import Retriever, compute_quantitative_score, compute_outcome_relevance_score
except ImportError:
    from rag.embeddings import EmbeddingManager
    from rag.vector_store import VectorStore, INDEX_DIR
    from rag.retriever import Retriever, compute_quantitative_score, compute_outcome_relevance_score


CHUNKS_FILE = Path("data/processed/chunks/chunks.jsonl")

AUDIT_QUERIES = [
    ("Q1", "Does empagliflozin increase the risk of genital infection compared with placebo?", "genital infection"),
    ("Q2", "What adverse events involving genital infections were reported with empagliflozin?", "genital infection"),
    ("Q3", "What was the cardiovascular outcome of empagliflozin?", "cardiovascular outcome"),
    ("Q4", "What were the common adverse events in patients treated with empagliflozin?", "adverse events")
]


def load_chunks() -> List[Dict[str, Any]]:
    """Load validated chunks from jsonl file."""
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Chunks file missing: {CHUNKS_FILE}")

    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def build_vector_store() -> VectorStore:
    """Build, populate, and save FAISS vector store from chunks."""
    chunks = load_chunks()
    embedder = EmbeddingManager()
    texts = [c["text"] for c in chunks]
    vectors = embedder.embed_batch(texts)
    v_store = VectorStore(dimension=embedder.embedding_dimension)
    v_store.add_chunks(vectors, chunks)
    v_store.save(INDEX_DIR)
    return v_store


class TestTask19ARetrieval(unittest.TestCase):
    """
    Task 19A Regression Tests verifying general retrieval improvements across queries A-F.
    """

    @classmethod
    def setUpClass(cls):
        cls.retriever = Retriever()

    def test_a_drug_adverse_event_comparator_question(self):
        """TEST A: Drug + adverse event + comparator question retrieves target EMP-005 chunk in candidate pool."""
        query = "Does empagliflozin increase the risk of genital infection compared with placebo?"
        results = self.retriever.retrieve(query, outcome="genital infection", top_k=25)
        
        chunk_ids = [r["chunk_id"] for r in results]
        self.assertIn("EMP-005_CHK_007", chunk_ids, "EMP-005_CHK_007 must appear in candidate pool for target question")
        
        # Verify provenance preservation
        target_chunk = next(r for r in results if r["chunk_id"] == "EMP-005_CHK_007")
        self.assertEqual(target_chunk["study_id"], "EMP-005")
        self.assertTrue(target_chunk["source_file"].endswith(".pdf") or "EMP-005" in target_chunk["source_file"])
        self.assertEqual(target_chunk["page_start"], 10)
        self.assertEqual(target_chunk["page_end"], 11)

    def test_b_another_adverse_event_question(self):
        """TEST B: Another adverse-event question retrieves valid genital infection safety evidence."""
        query = "What adverse events involving genital infections were reported with empagliflozin?"
        results = self.retriever.retrieve(query, outcome="genital infection", top_k=10)
        self.assertEqual(len(results), 10)
        
        relevant = [r for r in results if "genital" in r["text"].lower() or "infection" in r["text"].lower()]
        self.assertGreaterEqual(len(relevant), 8, "Expected at least 8 relevant chunks in top 10")

    def test_c_cardiovascular_outcome_question(self):
        """TEST C: Cardiovascular outcome question retrieves primary CV outcome chunks."""
        query = "What was the cardiovascular outcome of empagliflozin?"
        results = self.retriever.retrieve(query, outcome="cardiovascular outcome", top_k=10)
        self.assertEqual(len(results), 10)
        
        relevant = [r for r in results if "cardiovascular" in r["text"].lower() or "mace" in r["text"].lower() or "death" in r["text"].lower()]
        self.assertGreaterEqual(len(relevant), 8, "Expected CV outcome chunks in top 10")

    def test_d_answer_not_present_question(self):
        """TEST D: Question whose answer is not present in corpus returns top-k without crashing or hardcoding fake results."""
        query = "Does empagliflozin increase the risk of hepatotoxicity compared with placebo?"
        results = self.retriever.retrieve(query, outcome="hepatotoxicity", top_k=5)
        self.assertEqual(len(results), 5)

    def test_e_quantitative_table_query_another_study(self):
        """TEST E: Quantitative-table query from another study retrieves study table chunks naturally."""
        query = "What was the incidence of urinary tract infections in EMP-001?"
        results = self.retriever.retrieve(query, outcome="urinary tract infection", top_k=10)
        self.assertEqual(len(results), 10)
        
        emp001_chunks = [r for r in results if r["study_id"] == "EMP-001"]
        self.assertGreaterEqual(len(emp001_chunks), 1, "EMP-001 chunks must rank high for EMP-001 query")


    def test_f_unrelated_query(self):
        """TEST F: An unrelated query does not force EMP-005 merely because it is a primary RCT."""
        query = "What is the molecular mechanism of action of metformin in hepatocytes?"
        results = self.retriever.retrieve(query, top_k=5)
        self.assertEqual(len(results), 5)
        self.assertNotIn("EMP-005_CHK_007", [r["chunk_id"] for r in results])

    def test_q1_q4_audit_precision(self):
        """Audit Precision@5 on existing Q1-Q4 queries."""
        p5_scores = []
        for code, query, outcome in AUDIT_QUERIES:
            results = self.retriever.retrieve(query, outcome=outcome, top_k=5)
            self.assertEqual(len(results), 5)
            
            relevant_count = 0
            for r in results:
                txt = r["text"].lower()
                if "cardiovascular" in outcome.lower():
                    rel = "cardiovascular" in txt or "mace" in txt or "death" in txt or "heart" in txt
                elif "genital" in outcome.lower():
                    rel = "genital" in txt or "infection" in txt or "mycotic" in txt
                else:
                    rel = "adverse" in txt or "safety" in txt or "event" in txt
                if rel:
                    relevant_count += 1
            p5 = relevant_count / 5.0
            p5_scores.append(p5)

        avg_p5 = sum(p5_scores) / len(p5_scores)
        self.assertGreaterEqual(avg_p5, 0.95, f"Average Precision@5 fallen below 0.95: {avg_p5}")

    def test_no_hardcoded_emp005_logic(self):
        """Verify no hard-coded EMP-005 string exists in retriever.py implementation."""
        retriever_file = Path(__file__).parent / "retriever.py"
        with open(retriever_file, "r", encoding="utf-8") as f:
            code = f.read()

        self.assertNotIn("EMP-005", code, "Hard-coded 'EMP-005' found in retriever.py!")
        self.assertNotIn("EMP-005_CHK_007", code, "Hard-coded 'EMP-005_CHK_007' found in retriever.py!")


if __name__ == "__main__":
    unittest.main()
