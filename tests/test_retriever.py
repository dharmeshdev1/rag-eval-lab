import json
from pathlib import Path

import pytest

from ragkit.cli import run_eval
from ragkit.retriever import Retriever

DATA = Path(__file__).resolve().parent.parent / "data"


def test_search_returns_distinct_documents():
    r = Retriever(chunk_size=5, overlap=1)
    r.add_documents({"a": "alpha beta gamma delta epsilon zeta eta theta", "b": "unrelated words"})
    hits = r.search("alpha theta", k=5, mode="bm25")
    assert hits == ["a"]


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        Retriever().search("q", mode="magic")  # type: ignore[arg-type]


def test_bundled_dataset_is_well_formed():
    docs = {json.loads(line)["id"] for line in (DATA / "corpus.jsonl").read_text().splitlines()}
    for line in (DATA / "queries.jsonl").read_text().splitlines():
        assert set(json.loads(line)["relevant"]) <= docs


def test_eval_runs_end_to_end_and_hybrid_is_competitive():
    results = run_eval(DATA / "corpus.jsonl", DATA / "queries.jsonl", k=5)
    assert set(results) == {"bm25", "vector", "hybrid"}
    best_single = max(results["bm25"]["ndcg@5"], results["vector"]["ndcg@5"])
    assert results["hybrid"]["ndcg@5"] >= best_single - 0.05
