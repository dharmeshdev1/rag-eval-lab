from __future__ import annotations

import math
from collections import Counter

from ragkit.chunking import Chunk
from ragkit.text import tokenize


class BM25Index:
    """Okapi BM25 over chunks, with the non-negative idf variant used by Lucene."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._tf: list[Counter[str]] = []
        self._len: list[int] = []
        self._df: Counter[str] = Counter()

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            tf = Counter(tokens)
            self._chunks.append(chunk)
            self._tf.append(tf)
            self._len.append(len(tokens))
            self._df.update(tf.keys())

    def search(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        n = len(self._chunks)
        if n == 0:
            return []
        avg_len = sum(self._len) / n or 1.0
        terms = set(tokenize(query))
        scored: list[tuple[Chunk, float]] = []
        for chunk, tf, length in zip(self._chunks, self._tf, self._len, strict=True):
            score = 0.0
            for term in terms:
                freq = tf.get(term)
                if not freq:
                    continue
                df = self._df[term]
                idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
                norm = freq + self.k1 * (1.0 - self.b + self.b * length / avg_len)
                score += idf * freq * (self.k1 + 1.0) / norm
            if score > 0.0:
                scored.append((chunk, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0].chunk_id))
        return scored[:k]
