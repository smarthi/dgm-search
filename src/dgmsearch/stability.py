"""Stability energy V(agent).

PLACEHOLDER. This is an operational proxy, not the Lyapunov-RAG formalization.
V is the degradation of retrieval quality under bounded input perturbation:

    V = mean_p(drop_p) + lam * max_p(drop_p),   drop_p = max(0, s_base - s_p)

over perturbation (kind, seed) cells p. The gate in gates.py only needs a scalar that
should be non-increasing along an accepted lineage (a Lyapunov-style descent condition).
To test the real formalization, replace `energy` below and keep the signature.
"""
from __future__ import annotations


def energy(base: float, probe_scores: dict[str, float], lam: float = 0.5) -> float:
    if not probe_scores:
        return 0.0
    drops = [max(0.0, base - s) for s in probe_scores.values()]
    return sum(drops) / len(drops) + lam * max(drops)
