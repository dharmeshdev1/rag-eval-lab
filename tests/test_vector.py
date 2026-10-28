import pytest

from ragkit.chunking import Chunk
from ragkit.vector import VectorIndex, cosine, embed


def test_embedding_is_unit_length():
    vec = embed("retrieval augmented generation")
    assert sum(v * v for v in vec.values()) == pytest.approx(1.0)


def test_embedding_is_deterministic():
    assert embed("hello world") == embed("hello world")


def test_empty_text_gives_empty_vector():
    assert embed("the of and") == {}


def test_morphological_variants_are_similar_but_not_identical():
    sim = cosine(embed("retrieval"), embed("retrieve"))
    assert 0.2 < sim < 1.0


def test_unrelated_words_score_lower_than_variants():
    variant = cosine(embed("retrieval"), embed("retrieve"))
    unrelated = cosine(embed("retrieval"), embed("banana"))
    assert variant > unrelated


def test_search_finds_morphological_match():
    idx = VectorIndex()
    idx.add([Chunk("a", "a#0", "ranking documents"), Chunk("b", "b#0", "cooking pasta")])
    assert idx.search("rank document")[0][0].doc_id == "a"
