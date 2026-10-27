"""Deterministic query perturbations used as stability probes."""
from __future__ import annotations

import random

KINDS = ("typo", "drop", "shuffle")


def perturb_query(text: str, kind: str, rng: random.Random, intensity: float) -> str:
    words = text.split()
    if len(words) < 2:
        return text
    if kind == "typo":
        out = []
        for w in words:
            if len(w) > 3 and rng.random() < intensity:
                i = rng.randrange(len(w) - 1)
                w = w[:i] + w[i + 1] + w[i] + w[i + 2 :]
            out.append(w)
        return " ".join(out)
    if kind == "drop":
        keep = [w for w in words if rng.random() >= intensity]
        return " ".join(keep) if keep else rng.choice(words)
    if kind == "shuffle":
        out = words[:]
        rng.shuffle(out)
        return " ".join(out)
    raise ValueError(kind)


def build_probe_queries(queries: dict[str, str], seeds, intensity: float) -> dict[str, str]:
    """Probe ids look like '<qid>::<kind>::<seed>'."""
    probes = {}
    for kind in KINDS:
        for s in seeds:
            for qid, text in queries.items():
                rng = random.Random(f"{kind}|{s}|{qid}")
                probes[f"{qid}::{kind}::{s}"] = perturb_query(text, kind, rng, intensity)
    return probes
