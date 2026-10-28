import math

import pytest

from ragkit.metrics import evaluate_rankings, ndcg_at_k, recall_at_k, reciprocal_rank


def test_recall_counts_relevant_found_in_top_k():
    assert recall_at_k(["a", "x", "b"], {"a", "b"}, k=2) == 0.5
    assert recall_at_k(["a", "x", "b"], {"a", "b"}, k=3) == 1.0


def test_recall_with_no_relevant_docs_is_zero():
    assert recall_at_k(["a"], set(), k=1) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x"], {"a"}) == 0.0


def test_ndcg_is_one_for_ideal_order():
    assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, k=3) == pytest.approx(1.0)


def test_ndcg_known_value():
    expected = (1 / math.log2(3)) / 1.0  # single relevant doc at rank 2, ideal is rank 1
    assert ndcg_at_k(["x", "a"], {"a"}, k=2) == pytest.approx(expected)


def test_ndcg_penalises_late_hits():
    assert ndcg_at_k(["x", "a"], {"a"}, 2) < ndcg_at_k(["a", "x"], {"a"}, 2)


def test_missing_query_counts_as_zero_not_skipped():
    scores = evaluate_rankings({"q1": ["a"]}, {"q1": {"a"}, "q2": {"b"}}, k=1)
    assert scores["mrr"] == 0.5


def test_empty_qrels_rejected():
    with pytest.raises(ValueError):
        evaluate_rankings({}, {})
