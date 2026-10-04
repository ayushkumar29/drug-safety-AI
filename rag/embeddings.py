"""
Embedding Manager module using sentence-transformers.
Provides a lightweight model for text and batch embeddings.
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


# Default model: lightweight, high-performance, 384-dimensional dense vectors
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingManager:
    """
    Embedding manager wrapping a SentenceTransformer model for single and batch text embeddings.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    @property
    def embedding_dimension(self) -> int:
        """Return the vector dimensionality of the embedding model."""
        if hasattr(self.model, "get_embedding_dimension"):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Embed a single text string into a numpy float32 vector.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text string.")
        
        vector = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        return vector.astype(np.float32)

    def embed_batch(self, texts: List[str], normalize: bool = True, batch_size: int = 32) -> np.ndarray:
        """
        Embed a list of text strings into a 2D numpy float32 array (N, dimension).
        """
        if not texts:
            raise ValueError("Text batch cannot be empty.")
        
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        return vectors.astype(np.float32)
