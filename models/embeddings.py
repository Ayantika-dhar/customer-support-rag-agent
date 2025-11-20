# models/embeddings.py

from typing import List
import os
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.config import get_config


class EmbeddingClient:
    """
    Thin wrapper around a SentenceTransformer model.

    Used for:
    - embed_documents(list[str]) -> list[list[float]]
    - embed_query(str) -> list[float]
    """

    def __init__(self, model_name: str | None = None):
        config = get_config()
        self.model_name = model_name or config.get(
            "EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"
        )
        self.model = SentenceTransformer(self.model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Compute embeddings for multiple texts.
        Returns a list of embedding vectors (as Python lists of floats).
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine similarity works better
        )
        # Ensure we return a list of lists (not NumPy array)
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Convenience method for a single query string.
        """
        if not text:
            return []

        emb = self.embed_documents([text])[0]
        return emb
