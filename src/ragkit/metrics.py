from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: Collection[str]) -> float:
    for rank, doc in enumerate(ranked, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Binary-relevance nDCG with the standard log2 discount."""
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Collection[str]],
    k: int = 5,
) -> dict[str, float]:
    """Mean recall@k, MRR and nDCG@k over every query in ``qrels``.

    A query missing from ``rankings`` counts as an empty ranking rather than
    being skipped, so a system cannot raise its score by failing to answer.
    """
    if not qrels:
        raise ValueError("qrels must not be empty")
    n = len(qrels)
    recall = mrr = ndcg = 0.0
    for qid, relevant in qrels.items():
        ranked = rankings.get(qid, [])
        recall += recall_at_k(ranked, relevant, k)
        mrr += reciprocal_rank(ranked, relevant)
        ndcg += ndcg_at_k(ranked, relevant, k)
    return {f"recall@{k}": recall / n, "mrr": mrr / n, f"ndcg@{k}": ndcg / n}
