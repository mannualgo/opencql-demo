"""
OpenCQL Vector Store v2
- In-memory store with real cosine similarity (via sentence-transformers if available,
  else falls back to lightweight TF-IDF-style scoring)
- Interface designed to plug in Chroma / Pinecone / pgvector with minimal changes
"""

from __future__ import annotations
import math
import re
from collections import Counter
from typing import Any


# \u2500\u2500 Lightweight TF-IDF embedding fallback \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def _tfidf_vector(text: str, vocab: dict[str, int]) -> list[float]:
    tokens = _tokenize(text)
    counts = Counter(tokens)
    vec = [0.0] * len(vocab)
    for tok, cnt in counts.items():
        if tok in vocab:
            vec[vocab[tok]] = cnt
    return vec


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# \u2500\u2500 Try to load sentence-transformers for real embeddings \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

_ENCODER = None

def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        try:
            from sentence_transformers import SentenceTransformer
            _ENCODER = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            _ENCODER = "tfidf"
    return _ENCODER


# \u2500\u2500 VectorStore \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

class VectorStore:
    """
    In-memory vector store.
    Each document is a dict with at least {"text": str} plus optional metadata fields.
    """

    def __init__(self):
        self.documents: list[dict[str, Any]] = []
        self._embeddings: list[list[float]] = []
        self._vocab: dict[str, int] = {}  # used in TF-IDF mode

    # \u2500\u2500 Ingestion \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def add_documents(self, docs: list[dict[str, Any]]):
        """Add documents to the store. Each doc must have a 'text' key."""
        for doc in docs:
            self.documents.append(doc)
        self._rebuild_index()

    def add_document(self, doc: dict[str, Any]):
        self.documents.append(doc)
        self._rebuild_index()

    def _rebuild_index(self):
        enc = _get_encoder()
        texts = [d["text"] for d in self.documents]

        if enc == "tfidf":
            # Build vocabulary from all documents
            all_tokens = []
            for t in texts:
                all_tokens.extend(_tokenize(t))
            vocab_terms = list(set(all_tokens))
            self._vocab = {t: i for i, t in enumerate(vocab_terms)}
            self._embeddings = [_tfidf_vector(t, self._vocab) for t in texts]
        else:
            vecs = enc.encode(texts)
            self._embeddings = [list(map(float, v)) for v in vecs]

    def _embed_query(self, query: str) -> list[float]:
        enc = _get_encoder()
        if enc == "tfidf":
            return _tfidf_vector(query, self._vocab)
        else:
            return list(map(float, enc.encode([query])[0]))

    # \u2500\u2500 Retrieval \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
        filters: dict | None = None,
    ) -> list[tuple[dict, float]]:
        """
        Returns list of (doc, score) sorted by descending similarity.
        Optional metadata filters: {"domain": "Legal"} will only return docs
        where doc["domain"] == "Legal".
        """
        if not self.documents:
            return []

        q_vec = self._embed_query(query)
        scored = []
        for doc, emb in zip(self.documents, self._embeddings):
            score = _cosine(q_vec, emb)
            if score >= threshold:
                if filters:
                    if all(doc.get(k) == v for k, v in filters.items()):
                        scored.append((doc, score))
                else:
                    scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def search_by_field(
        self, field: str, value: Any
    ) -> list[dict]:
        """Exact match on a metadata field."""
        return [d for d in self.documents if d.get(field) == value]

    def get_partitions(self, field: str) -> list[Any]:
        """Return unique values for a metadata field (for AUTO partitioning)."""
        seen = set()
        result = []
        for doc in self.documents:
            val = doc.get(field)
            if val is not None and val not in seen:
                seen.add(val)
                result.append(val)
        return result

    def count(self) -> int:
        return len(self.documents)

    def clear(self):
        self.documents.clear()
        self._embeddings.clear()
        self._vocab.clear()


# \u2500\u2500 Named source registry \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

class SourceRegistry:
    """Holds named VectorStore instances (e.g. 'docs.product', 'history.customer')."""

    def __init__(self):
        self._stores: dict[str, VectorStore] = {}

    def register(self, name: str, store: VectorStore):
        self._stores[name] = store

    def get(self, name: str) -> VectorStore | None:
        return self._stores.get(name)

    def get_or_create(self, name: str) -> VectorStore:
        if name not in self._stores:
            self._stores[name] = VectorStore()
        return self._stores[name]

    def list_sources(self) -> list[str]:
        return list(self._stores.keys())
