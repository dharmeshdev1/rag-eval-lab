# rag-eval-lab

[![ci](https://github.com/dharmeshdev1/rag-eval-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/dharmeshdev1/rag-eval-lab/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

A small retrieval library and evaluation harness. It implements BM25, a hashed
character n-gram vector retriever and reciprocal rank fusion from scratch, then
scores each strategy with recall@k, MRR and nDCG@k so you can compare them on
your own data before choosing one.

There are no runtime dependencies. Everything is plain Python, typed and tested,
so each piece is short enough to read in one sitting.

## Why this exists

Most RAG failures start at retrieval, not generation. A reranker cannot recover
a document the first stage never returned, and a language model cannot cite a
passage it was never shown. This project makes the retrieval stage measurable:
swap a chunk size or a fusion strategy, rerun one command and see what moved.

## Architecture

```
                    +-----------+
 documents -------> |  chunking | overlapping word windows
                    +-----+-----+
                          |
              +-----------+-----------+
              v                       v
        +-----------+          +--------------+
        |   BM25    |          | hashed char  |
        |  (lexical)|          | n-gram vector|
        +-----+-----+          +------+-------+
              |  ranked docs          |  ranked docs
              +-----------+-----------+
                          v
                +--------------------+
                | reciprocal rank    |
                | fusion (hybrid)    |
                +---------+----------+
                          v
                +--------------------+
                | metrics: recall@k, |
                | MRR, nDCG@k        |
                +--------------------+
```

## Quickstart

```bash
git clone https://github.com/dharmeshdev1/rag-eval-lab
cd rag-eval-lab
pip install -e ".[dev]"
pytest -q
ragkit eval
```

```python
from ragkit import Retriever

retriever = Retriever(chunk_size=120, overlap=30)
retriever.add_documents({"a": "BM25 scores exact keyword matches.", "b": "Vectors find paraphrases."})
retriever.search("keyword scoring", k=3, mode="hybrid")  # ["a", "b"]
```

Evaluate on your own data with JSONL files. A corpus row is `{"id", "text"}` and a
query row is `{"id", "query", "relevant": [doc ids]}`:

```bash
ragkit eval --corpus my_corpus.jsonl --queries my_queries.jsonl -k 10
```

## Design notes

- **BM25 uses the non-negative idf variant** (`log(1 + (N - df + 0.5) / (df + 0.5))`)
  so a term present in every document contributes zero instead of a negative score.
- **The vector retriever hashes character trigrams**, which gives partial credit to
  morphological variants such as "retrieve" and "retrieval" that exact-match BM25
  scores as zero. It uses `zlib.crc32` rather than Python's `hash()` so results are
  identical across processes. It is not a learned embedding model, and it will not
  match true paraphrases that share no characters.
- **Fusion uses RRF** because it needs no score calibration. BM25 scores and cosine
  similarities live on different scales, and RRF only looks at ranks.
- **Metrics treat a missing query as an empty ranking.** A system cannot raise its
  mean score by skipping the queries it cannot answer.
- **Ties break by chunk id**, so rankings are deterministic and tests are stable.

## Results on the bundled dataset

```
mode        recall@5         mrr      ndcg@5
--------------------------------------------
bm25           0.958       1.000       0.968
vector         1.000       1.000       0.990
hybrid         0.958       1.000       0.968
```

Read this with care. The bundled corpus has 12 documents and 12 queries, so it
works as a smoke test that the pipeline runs end to end, not as a benchmark. All
three strategies are close to the ceiling, and a difference of one missed document
moves recall by several points. The harness is meant to be pointed at a real corpus
with realistic queries, where the strategies separate.

## Limitations and next steps

- Swap the hashed vector index for a learned embedding model behind the same
  `search` interface.
- Add a cross encoder reranking stage over the fused candidate list.
- Support graded relevance labels in `ndcg_at_k`, which is binary today.
- Report per-query results and bootstrap confidence intervals, not only means.

## Development

```bash
ruff check .
pytest -q
```

CI runs lint, tests and the evaluation on Python 3.10 to 3.13.

## License

MIT
