from __future__ import annotations

from typing import Literal

from ragkit.bm25 import BM25Index
from ragkit.chunking import Chunk, chunk_words
from ragkit.fusion import reciprocal_rank_fusion
from ragkit.vector import VectorIndex

Mode = Literal["bm25", "vector", "hybrid"]


class Retriever:
    """Chunk documents once, then rank documents with BM25, vectors or both."""

    def __init__(self, chunk_size: int = 120, overlap: int = 30) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._bm25 = BM25Index()
        self._vector = VectorIndex()

    def add_documents(self, docs: dict[str, str]) -> None:
        chunks: list[Chunk] = []
        for doc_id, text in docs.items():
            chunks.extend(chunk_words(doc_id, text, self.chunk_size, self.overlap))
        self._bm25.add(chunks)
        self._vector.add(chunks)

    def search(self, query: str, k: int = 10, mode: Mode = "hybrid") -> list[str]:
        """Return up to ``k`` distinct document ids, best first."""
        depth = k * 5  # over-fetch chunks so de-duplication still leaves k documents
        if mode == "bm25":
            ranked = self._doc_ranking(self._bm25.search(query, depth))
        elif mode == "vector":
            ranked = self._doc_ranking(self._vector.search(query, depth))
        elif mode == "hybrid":
            lists = [
                self._doc_ranking(self._bm25.search(query, depth)),
                self._doc_ranking(self._vector.search(query, depth)),
            ]
            ranked = [doc for doc, _ in reciprocal_rank_fusion(lists)]
        else:
            raise ValueError(f"unknown mode: {mode!r}")
        return ranked[:k]

    @staticmethod
    def _doc_ranking(hits: list[tuple[Chunk, float]]) -> list[str]:
        seen: dict[str, None] = {}
        for chunk, _ in hits:
            seen.setdefault(chunk.doc_id, None)
        return list(seen)
