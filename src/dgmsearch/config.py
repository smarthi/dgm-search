from __future__ import annotations

from dataclasses import dataclass, field

ARMS = ("open", "score", "score_stab", "score_size")


@dataclass
class RunConfig:
    data_dir: str = "data/toy"
    out_dir: str = "runs"
    arm: str = "score"
    mutator: str = "rule"          # rule | llm
    iters: int = 40
    seed: int = 0
    k: int = 10
    # stability probes used for selection (dev) -- the final shifted eval uses different seeds/intensity
    probe_seeds: tuple = (0, 1)
    probe_intensity: float = 0.25
    shift_seeds: tuple = (100, 101)
    shift_intensity: float = 0.5
    lam: float = 0.5
    # gate parameters (identical across arms where shared)
    score_tol: float = 0.005       # base rule: child dev nDCG >= parent - score_tol
    energy_tol: float = 0.005      # score_stab: child V <= parent V + energy_tol
    open_floor: float = 0.9        # open: child dev nDCG >= open_floor * seed
    size_slack: float = 1.10       # score_size: child lines <= size_slack * parent lines
    timeout: float = 120.0
    mem_mb: int = 2048
    model: str = "claude-sonnet-5-5"
    extra: dict = field(default_factory=dict)
