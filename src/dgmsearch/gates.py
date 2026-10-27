"""Archive admission rules. Parent selection, mutator and budget are identical across arms;
only this function differs. That is the experimental manipulation.

  open        DGM-style: keep anything valid that is not much worse than the seed.
  score       child dev nDCG >= parent - score_tol.
  score_stab  `score` AND stability energy non-increasing (V_child <= V_parent + energy_tol).
  score_size  `score` AND code size <= size_slack * parent. Placebo regularizer: controls for
              "any extra constraint helps" versus "stability specifically helps".
"""
from __future__ import annotations

from .config import ARMS, RunConfig
from .evaluate import EvalResult


def admit(cfg: RunConfig, child: EvalResult, parent: EvalResult, seed: EvalResult) -> tuple[bool, str]:
    if cfg.arm not in ARMS:
        raise ValueError(f"unknown arm {cfg.arm}")
    if not child.valid:
        return False, "invalid"
    if cfg.arm == "open":
        ok = child.ndcg >= cfg.open_floor * seed.ndcg
        return ok, "ok" if ok else "below_floor"
    if child.ndcg < parent.ndcg - cfg.score_tol:
        return False, "score_drop"
    if cfg.arm == "score":
        return True, "ok"
    if cfg.arm == "score_stab":
        if child.energy > parent.energy + cfg.energy_tol:
            return False, "energy_up"
        return True, "ok"
    if cfg.arm == "score_size":
        if child.n_lines > cfg.size_slack * parent.n_lines:
            return False, "too_big"
        return True, "ok"
    raise AssertionError
