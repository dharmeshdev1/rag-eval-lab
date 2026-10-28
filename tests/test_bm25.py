from ragkit.bm25 import BM25Index
from ragkit.chunking import Chunk


def _index(*texts):
    idx = BM25Index()
    idx.add([Chunk(f"d{i}", f"d{i}#0", t) for i, t in enumerate(texts)])
    return idx


def test_exact_term_ranks_first():
    idx = _index("cats and dogs", "quantum chromodynamics lecture", "weather report")
    hits = idx.search("quantum chromodynamics")
    assert hits[0][0].doc_id == "d1"


def test_rare_term_outweighs_common_term():
    idx = _index("model model model data", "model data zebra", "model data", "model data")
    assert idx.search("model zebra")[0][0].doc_id == "d1"


def test_no_overlap_returns_nothing():
    assert _index("alpha beta").search("gamma") == []


def test_empty_index():
    assert BM25Index().search("anything") == []


def test_scores_are_non_negative_even_for_ubiquitous_terms():
    idx = _index("model", "model", "model")
    assert all(score >= 0 for _, score in idx.search("model"))


def test_ties_break_deterministically():
    idx = _index("same words", "same words")
    assert [c.doc_id for c, _ in idx.search("same words")] == ["d0", "d1"]
