import pytest

from ragkit.fusion import reciprocal_rank_fusion


def test_item_in_both_lists_beats_item_in_one():
    fused = reciprocal_rank_fusion([["a", "b"], ["c", "b"]])
    assert fused[0][0] == "b"


def test_scores_match_formula():
    fused = dict(reciprocal_rank_fusion([["a", "b"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62)


def test_empty_input():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_order_of_lists_does_not_matter():
    a = reciprocal_rank_fusion([["x", "y"], ["y", "z"]])
    b = reciprocal_rank_fusion([["y", "z"], ["x", "y"]])
    assert a == b
