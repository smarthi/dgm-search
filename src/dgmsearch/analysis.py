"""Aggregate runs/*/final.json into the pre-registered comparison (see docs/PREREGISTRATION.md)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ["dev_ndcg", "test_ndcg", "gap", "shift_drop", "lineage_energy_increase_rate"]


def _boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, len(x))).mean(1)
    return np.percentile(means, [2.5, 97.5])


def _boot_diff(a, b, n=10000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    d = rng.choice(a, (n, len(a))).mean(1) - rng.choice(b, (n, len(b))).mean(1)
    return float(a.mean() - b.mean()), np.percentile(d, [2.5, 97.5])


def analyze(runs_dir: str, mutator: str | None = None) -> str:
    by_arm = defaultdict(list)
    for p in sorted(Path(runs_dir).glob("*/final.json")):
        r = json.loads(p.read_text())
        if mutator and r["mutator"] != mutator:
            continue
        by_arm[r["arm"]].append(r)
    if not by_arm:
        return "no runs found"
    out = ["| arm | n | " + " | ".join(METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    for arm, rs in by_arm.items():
        cells = []
        for m in METRICS:
            v = [r[m] for r in rs]
            ci = _boot_ci(v) if len(v) > 1 else (float("nan"),) * 2
            cells.append(f"{np.mean(v):.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
        out.append(f"| {arm} | {len(rs)} | " + " | ".join(cells) + " |")
    seed_t = [r["seed_test_ndcg"] for rs in by_arm.values() for r in rs]
    out.append(f"\nseed (BM25) test nDCG@10: {np.mean(seed_t):.3f}")
    if "score_stab" in by_arm and "score" in by_arm and len(by_arm["score"]) > 1:
        out.append("\nPrimary contrasts (score_stab - score), mean diff [95% bootstrap CI over seeds]:")
        for m in ("test_ndcg", "gap", "shift_drop"):
            d, ci = _boot_diff([r[m] for r in by_arm["score_stab"]], [r[m] for r in by_arm["score"]])
            out.append(f"  {m}: {d:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    return "\n".join(out)
