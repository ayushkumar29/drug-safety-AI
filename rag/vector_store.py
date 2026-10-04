"""
FAISS Vector Store module.
Handles index creation, vector + metadata storage, local persistence, and similarity search.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import faiss
import numpy as np


INDEX_DIR = Path("data/processed/vector_store")
INDEX_FILE = INDEX_DIR / "index.faiss"
METADATA_FILE = INDEX_DIR / "metadata.json"


class VectorStore:
    """
    FAISS-backed Vector Store supporting cosine similarity / inner product search
    and metadata association.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        # Use IndexFlatIP for cosine similarity on normalized vectors
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: List[Dict[str, Any]] = []

    @property
    def total_vectors(self) -> int:
        """Return total number of vectors in FAISS index."""
        return self.index.ntotal

    def add_chunks(self, vectors: np.ndarray, chunk_metadatas: List[Dict[str, Any]]) -> None:
        """
        Add embedding vectors and associated metadata to the vector store.
        """
        if len(vectors) != len(chunk_metadatas):
            raise ValueError(f"Vector count ({len(vectors)}) does not match metadata count ({len(chunk_metadatas)})")

        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension ({vectors.shape[1]}) does not match index dimension ({self.dimension})")

        # Ensure float32 format
        vectors = vectors.astype(np.float32)

        # Normalize vectors for cosine similarity if using IndexFlatIP
        faiss.normalize_L2(vectors)

        self.index.add(vectors)
        self.metadata.extend(chunk_metadatas)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search vector store for top_k most similar vectors.

        Returns list of result dicts containing metadata and similarity_score.
        """
        if self.index.ntotal == 0:
            return []

        # Format query vector
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        query_vector = query_vector.astype(np.float32)
        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = dict(self.metadata[idx])
            meta["similarity_score"] = float(score)
            results.append(meta)

        return results

    def save(self, output_dir: Path = INDEX_DIR) -> None:
        """Save FAISS index and metadata to local disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(output_dir / "index.faiss"))
        with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, index_dir: Path = INDEX_DIR) -> "VectorStore":
        """Load VectorStore from local disk."""
        index_path = index_dir / "index.faiss"
        meta_path = index_dir / "metadata.json"

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Vector store files missing in {index_dir}")

        faiss_index = faiss.read_index(str(index_path))
        dimension = faiss_index.d

        store = cls(dimension=dimension)
        store.index = faiss_index

        with open(meta_path, "r", encoding="utf-8") as f:
            store.metadata = json.load(f)

        return store
