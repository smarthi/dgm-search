"""Dataset format and a synthetic toy generator.

On-disk layout (one directory per dataset):
    corpus.jsonl  {"id": str, "text": str}
    dev.jsonl     {"id": str, "text": str, "relevant": {doc_id: grade}}   <- used by the evolve loop
    test.jsonl    same schema                                           <- never seen by the loop

The toy generator exists to exercise the machinery. Results on it mean nothing.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Dataset:
    corpus: dict[str, str]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]


def _read(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write(path: Path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_dataset(root: str | Path, split: str) -> Dataset:
    root = Path(root)
    corpus = {r["id"]: r["text"] for r in _read(root / "corpus.jsonl")}
    queries, qrels = {}, {}
    for r in _read(root / f"{split}.jsonl"):
        queries[r["id"]] = r["text"]
        qrels[r["id"]] = {str(d): int(g) for d, g in r["relevant"].items()}
    return Dataset(corpus, queries, qrels)


def make_toy(root: str | Path, n_docs=300, n_topics=15, n_dev=40, n_test=40, seed=0) -> None:
    """Synthetic topical corpus. Each query is a few words from one doc, plus
    one same-topic word and one off-topic distractor, so plain BM25 is good but not perfect."""
    rng = random.Random(seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    per_topic = 40
    topic_words = [[f"t{t}w{i}" for i in range(per_topic)] for t in range(n_topics)]
    docs, doc_topic = {}, {}
    for d in range(n_docs):
        t = rng.randrange(n_topics)
        words = []
        for _ in range(30):
            if rng.random() < 0.7:
                words.append(rng.choice(topic_words[t]))
            else:
                words.append(rng.choice(topic_words[rng.randrange(n_topics)]))
        docs[f"d{d}"] = " ".join(words)
        doc_topic[f"d{d}"] = t

    def make_queries(n, prefix):
        rows = []
        for i in range(n):
            did = rng.choice(list(docs))
            toks = docs[did].split()
            q = rng.sample(toks, 4)
            q.append(rng.choice(topic_words[doc_topic[did]]))
            q.append(rng.choice(topic_words[rng.randrange(n_topics)]))
            rng.shuffle(q)
            rows.append({"id": f"{prefix}{i}", "text": " ".join(q), "relevant": {did: 2}})
        return rows

    _write(root / "corpus.jsonl", ({"id": k, "text": v} for k, v in docs.items()))
    _write(root / "dev.jsonl", make_queries(n_dev, "dq"))
    _write(root / "test.jsonl", make_queries(n_test, "tq"))
