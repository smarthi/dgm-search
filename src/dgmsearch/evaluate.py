from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

from .config import RunConfig
from .data import Dataset
from .metrics import ndcg_at_k, recall_at_k
from .perturb import build_probe_queries
from .sandbox import run_candidate
from .stability import energy


@dataclass
class EvalResult:
    valid: bool
    error: str | None = None
    ndcg: float = 0.0
    recall: float = 0.0
    energy: float = 0.0
    probe_mean: float = 0.0
    worst: list = field(default_factory=list)   # [(qid, ndcg)] lowest-scoring base queries
    seconds: float = 0.0
    n_lines: int = 0


def evaluate(path: str, ds: Dataset, cfg: RunConfig, probe_seeds, intensity) -> EvalResult:
    t0 = time.time()
    code = open(path, encoding="utf-8").read()
    n_lines = code.count("\n") + 1
    probes = build_probe_queries(ds.queries, probe_seeds, intensity)
    res = run_candidate(path, ds.corpus, {**ds.queries, **probes}, cfg.k, cfg.timeout, cfg.mem_mb)
    if res.error:
        return EvalResult(False, error=res.error, seconds=time.time() - t0, n_lines=n_lines)
    run = res.run
    known = set(ds.corpus)
    for qid in list(ds.queries) + list(probes):
        ranked = run.get(qid)
        if not isinstance(ranked, list) or any(d not in known for d in ranked):
            return EvalResult(False, error=f"invalid ranking for {qid}", seconds=time.time() - t0, n_lines=n_lines)

    def ranked_of(qid):
        seen, out = set(), []
        for d in run[qid]:
            if d not in seen:
                seen.add(d)
                out.append(d)
        return out

    per_q = {q: ndcg_at_k(ranked_of(q), ds.qrels[q], cfg.k) for q in ds.queries}
    base = sum(per_q.values()) / len(per_q)
    recall = sum(recall_at_k(ranked_of(q), ds.qrels[q], cfg.k) for q in ds.queries) / len(ds.queries)

    cells = defaultdict(list)
    for pid in probes:
        qid, kind, seed = pid.split("::")
        cells[f"{kind}|{seed}"].append(ndcg_at_k(ranked_of(pid), ds.qrels[qid], cfg.k))
    cell_scores = {c: sum(v) / len(v) for c, v in cells.items()}
    probe_mean = sum(cell_scores.values()) / len(cell_scores) if cell_scores else base
    worst = sorted(per_q.items(), key=lambda kv: kv[1])[:5]
    return EvalResult(True, None, base, recall, energy(base, cell_scores, cfg.lam), probe_mean,
                      worst, time.time() - t0, n_lines)
