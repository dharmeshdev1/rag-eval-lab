from __future__ import annotations

import math
import zlib
from collections import Counter

from ragkit.chunking import Chunk
from ragkit.text import tokenize

SparseVec = dict[int, float]


def _ngrams(token: str, n: int) -> list[str]:
    padded = f"<{token}>"
    if len(padded) <= n:
        return [padded]
    return [padded[i : i + n] for i in range(len(padded) - n + 1)]


def embed(text: str, dim: int = 4096, n: int = 3) -> SparseVec:
    """Hashed character n-gram embedding, L2 normalised.

    Character n-grams give partial credit to morphological variants
    ("retrieve" and "retrieval"), which exact-match BM25 scores as zero.
    ``zlib.crc32`` keeps the hashing deterministic across processes.
    """
    counts: Counter[int] = Counter()
    for token in tokenize(text):
        for gram in _ngrams(token, n):
            counts[zlib.crc32(gram.encode()) % dim] += 1
    vec = {i: 1.0 + math.log(c) for i, c in counts.items()}
    norm = math.sqrt(sum(v * v for v in vec.values()))
    return {i: v / norm for i, v in vec.items()} if norm else {}


def cosine(a: SparseVec, b: SparseVec) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(i, 0.0) for i, v in a.items())


class VectorIndex:
    def __init__(self, dim: int = 4096, n: int = 3) -> None:
        self.dim = dim
        self.n = n
        self._chunks: list[Chunk] = []
        self._vecs: list[SparseVec] = []

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._chunks.append(chunk)
            self._vecs.append(embed(chunk.text, self.dim, self.n))

    def search(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        q = embed(query, self.dim, self.n)
        scored = [
            (chunk, cosine(q, vec)) for chunk, vec in zip(self._chunks, self._vecs, strict=True)
        ]
        scored = [pair for pair in scored if pair[1] > 0.0]
        scored.sort(key=lambda pair: (-pair[1], pair[0].chunk_id))
        return scored[:k]
