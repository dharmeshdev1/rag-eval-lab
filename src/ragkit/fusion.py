from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TypeVar

T = TypeVar("T", bound=Hashable)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[T]], k: int = 60
) -> list[tuple[T, float]]:
    """Fuse ranked lists with RRF (Cormack et al., 2009).

    Each item scores ``sum(1 / (k + rank))`` over the lists it appears in, with
    ranks starting at 1. It needs no score calibration between retrievers, which
    is why it is a safe default for combining lexical and vector search.
    """
    scores: dict[T, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], str(pair[0])))
