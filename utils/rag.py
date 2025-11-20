# utils/rag.py

import os
from typing import List, Dict, Tuple

import numpy as np
from models.embeddings import EmbeddingClient


def load_documents(docs_dir: str) -> List[Dict]:
    """
    Load all .txt files from docs_dir.

    Returns a list of dicts:
    [
      {"id": "faq.txt", "text": "...full text...", "source": "faq.txt"},
      ...
    ]
    """
    docs: List[Dict] = []

    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    for fname in os.listdir(docs_dir):
        if not fname.lower().endswith(".txt"):
            continue

        path = os.path.join(docs_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            # Skip files that can’t be read
            print(f"[load_documents] Failed to read {path}: {e}")
            continue

        docs.append(
            {
                "id": fname,
                "text": text,
                "source": fname,
            }
        )

    return docs


def chunk_text(
    text: str, chunk_size: int = 800, overlap: int = 200
) -> List[str]:
    """
    Simple character-based chunking.

    Example: chunk_size=800, overlap=200
    text[0:800], text[600:1400], text[1200:2000], ...

    Returns a list of text chunks.
    """
    chunks: List[str] = []
    if not text:
        return chunks

    start = 0
    n = len(text)

    while start < n:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # move start forward but keep overlap with previous chunk
        start += chunk_size - overlap

    return chunks


def build_knowledge_base(
    docs_dir: str,
    embed_client: EmbeddingClient,
) -> Dict:
    """
    Build an in-memory 'vector store' from all docs in docs_dir.

    Returns a dict:
    {
        "embeddings": np.ndarray [num_chunks, dim],
        "chunks": [
            {"text": "...", "source": "faq.txt"},
            ...
        ]
    }
    """
    docs = load_documents(docs_dir)

    chunks: List[Dict] = []
    for d in docs:
        for ch in chunk_text(d["text"]):
            chunks.append(
                {
                    "text": ch,
                    "source": d["source"],
                }
            )

    if not chunks:
        raise ValueError(f"No chunks created from docs in: {docs_dir}")

    texts = [c["text"] for c in chunks]
    embeddings_list = embed_client.embed_documents(texts)  # List[List[float]]

    embeddings = np.array(embeddings_list, dtype="float32")

    vectorstore = {
        "embeddings": embeddings,
        "chunks": chunks,
    }
    return vectorstore


def cosine_similarity_matrix(
    query_vec: np.ndarray, doc_matrix: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between a query vector and each row in doc_matrix.
    Returns a 1D array of similarity scores.
    """
    # (n_dim,) -> (1, n_dim) for broadcasting
    q = query_vec.reshape(1, -1)
    # dot(q, docs^T) / (||q|| * ||docs||)
    dot = np.dot(doc_matrix, q.T).squeeze(-1)  # shape: (num_docs,)
    q_norm = np.linalg.norm(q)
    d_norm = np.linalg.norm(doc_matrix, axis=1) + 1e-8
    sims = dot / (q_norm * d_norm + 1e-8)
    return sims


def retrieve_relevant_chunks(
    query: str,
    embed_client: EmbeddingClient,
    vectorstore: Dict,
    top_k: int = 5,
) -> List[Dict]:
    """
    Given a user query and a vectorstore, return top_k most similar chunks.

    Returns:
    [
      {"text": "...chunk...", "source": "faq.txt", "score": 0.83},
      ...
    ]
    """
    if not query:
        return []

    if not vectorstore or "embeddings" not in vectorstore:
        raise ValueError("Vectorstore is empty or not built.")

    q_emb_list = embed_client.embed_query(query)
    if not q_emb_list:
        return []

    q_vec = np.array(q_emb_list, dtype="float32")
    doc_matrix = vectorstore["embeddings"]  # shape: (num_chunks, dim)

    sims = cosine_similarity_matrix(q_vec, doc_matrix)
    # Indices of top_k scores, sorted descending
    top_idx = np.argsort(-sims)[:top_k]

    results: List[Dict] = []
    for idx in top_idx:
        chunk = vectorstore["chunks"][int(idx)]
        score = float(sims[int(idx)])
        results.append(
            {
                "text": chunk["text"],
                "source": chunk["source"],
                "score": score,
            }
        )

    return results

