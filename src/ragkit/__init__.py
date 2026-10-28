"""ragkit: small retrieval primitives and an evaluation harness."""

from ragkit.chunking import Chunk, chunk_words
from ragkit.fusion import reciprocal_rank_fusion
from ragkit.metrics import evaluate_rankings, ndcg_at_k, recall_at_k, reciprocal_rank
from ragkit.retriever import Retriever

__all__ = [
    "Chunk",
    "Retriever",
    "chunk_words",
    "evaluate_rankings",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "reciprocal_rank_fusion",
]
