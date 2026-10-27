from __future__ import annotations

import math


def _dcg(gains) -> float:
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: list[str], qrel: dict[str, int], k: int) -> float:
    gains = [qrel.get(d, 0) for d in ranked[:k]]
    ideal = sorted(qrel.values(), reverse=True)[:k]
    idcg = _dcg(ideal)
    return _dcg(gains) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: list[str], qrel: dict[str, int], k: int) -> float:
    rel = {d for d, g in qrel.items() if g > 0}
    return len(rel & set(ranked[:k])) / len(rel) if rel else 0.0
