from __future__ import annotations

import argparse
import json
from pathlib import Path

from ragkit.metrics import evaluate_rankings
from ragkit.retriever import Retriever

MODES = ("bm25", "vector", "hybrid")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_eval(corpus: Path, queries: Path, k: int) -> dict[str, dict[str, float]]:
    docs = {row["id"]: row["text"] for row in _read_jsonl(corpus)}
    rows = _read_jsonl(queries)
    qrels = {row["id"]: set(row["relevant"]) for row in rows}

    retriever = Retriever()
    retriever.add_documents(docs)

    results: dict[str, dict[str, float]] = {}
    for mode in MODES:
        rankings = {row["id"]: retriever.search(row["query"], k=k, mode=mode) for row in rows}
        results[mode] = evaluate_rankings(rankings, qrels, k=k)
    return results


def format_table(results: dict[str, dict[str, float]]) -> str:
    metrics = list(next(iter(results.values())))
    header = f"{'mode':<8}" + "".join(f"{m:>12}" for m in metrics)
    lines = [header, "-" * len(header)]
    for mode, scores in results.items():
        lines.append(f"{mode:<8}" + "".join(f"{scores[m]:>12.3f}" for m in metrics))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ragkit", description="Retrieval evaluation harness")
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("eval", help="compare bm25, vector and hybrid retrieval")
    ev.add_argument("--corpus", type=Path, default=Path("data/corpus.jsonl"))
    ev.add_argument("--queries", type=Path, default=Path("data/queries.jsonl"))
    ev.add_argument("-k", type=int, default=5)
    ev.add_argument("--json", action="store_true", help="print raw JSON instead of a table")
    args = parser.parse_args(argv)

    results = run_eval(args.corpus, args.queries, args.k)
    print(json.dumps(results, indent=2) if args.json else format_table(results))


if __name__ == "__main__":
    main()
