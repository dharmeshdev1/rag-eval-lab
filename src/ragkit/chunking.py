from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: str
    text: str


def chunk_words(doc_id: str, text: str, size: int = 120, overlap: int = 30) -> list[Chunk]:
    """Split text into overlapping word windows.

    Windows advance by ``size - overlap`` words, so neighbouring chunks share
    ``overlap`` words of context. The last window is never shorter than needed
    to cover the remaining text.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")

    words = text.split()
    if not words:
        return []

    step = size - overlap
    chunks: list[Chunk] = []
    for n, start in enumerate(range(0, len(words), step)):
        window = words[start : start + size]
        chunks.append(Chunk(doc_id, f"{doc_id}#{n}", " ".join(window)))
        if start + size >= len(words):
            break
    return chunks
