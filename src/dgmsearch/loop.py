from __future__ import annotations

import importlib.resources as ir
import json
import random
import shutil
from dataclasses import asdict
from pathlib import Path

from .archive import Archive
from .config import RunConfig
from .data import load_dataset
from .evaluate import evaluate
from .gates import admit
from .mutators import make_mutator


def seed_code() -> str:
    return ir.files("dgmsearch").joinpath("seed/bm25_seed.py").read_text(encoding="utf-8")


def run_arm(cfg: RunConfig) -> dict:
    run_dir = Path(cfg.out_dir) / f"{cfg.arm}_{cfg.mutator}_s{cfg.seed}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    (run_dir / "rejected").mkdir(parents=True)
    rng = random.Random(cfg.seed)
    dev = load_dataset(cfg.data_dir, "dev")
    mutator = make_mutator(cfg.mutator, rng, dev, cfg.model)
    archive = Archive(run_dir)

    code0 = seed_code()
    p0 = run_dir / "seed.py"
    p0.write_text(code0, encoding="utf-8")
    seed_ev = evaluate(str(p0), dev, cfg, cfg.probe_seeds, cfg.probe_intensity)
    if not seed_ev.valid:
        raise RuntimeError(f"seed retriever failed: {seed_ev.error}")
    archive.add(code0, None, seed_ev, 0)

    log = open(run_dir / "log.jsonl", "w", encoding="utf-8")
    for it in range(1, cfg.iters + 1):
        parent = archive.select_parent(rng)
        parent_code = Path(parent.path).read_text(encoding="utf-8")
        try:
            child_code = mutator.mutate(parent_code, archive.evals[parent.id])
        except Exception as e:  # noqa: BLE001  (API errors etc. cost an iteration, not the run)
            log.write(json.dumps({"it": it, "parent": parent.id, "status": f"mutator_error: {e}"[:300]}) + "\n")
            continue
        if child_code is None:
            log.write(json.dumps({"it": it, "parent": parent.id, "status": "no_code"}) + "\n")
            continue
        tmp = run_dir / "rejected" / f"{it}.py"
        tmp.write_text(child_code, encoding="utf-8")
        ev = evaluate(str(tmp), dev, cfg, cfg.probe_seeds, cfg.probe_intensity)
        ok, reason = admit(cfg, ev, archive.evals[parent.id], seed_ev)
        rec = {"it": it, "parent": parent.id, "admitted": ok, "reason": reason, "valid": ev.valid,
               "ndcg": ev.ndcg, "energy": ev.energy, "error": ev.error}
        if ok:
            node = archive.add(child_code, parent, ev, it)
            rec["node"] = node.id
            tmp.unlink()
        log.write(json.dumps(rec) + "\n")
        log.flush()
    log.close()

    best = archive.best()
    final = final_eval(cfg, run_dir, archive, best, str(p0))
    (run_dir / "final.json").write_text(json.dumps(final, indent=2))
    return final


def final_eval(cfg: RunConfig, run_dir: Path, archive: Archive, best, seed_path: str) -> dict:
    """Touches the held-out test split exactly once per run, after selection is finished.
    Shift probes use different seeds and higher intensity than the in-loop probes."""
    test = load_dataset(cfg.data_dir, "test")
    best_t = evaluate(best.path, test, cfg, cfg.shift_seeds, cfg.shift_intensity)
    seed_t = evaluate(seed_path, test, cfg, cfg.shift_seeds, cfg.shift_intensity)
    inc, n_edges = archive.lineage_energy_increase_rate()
    return {
        "arm": cfg.arm, "mutator": cfg.mutator, "seed": cfg.seed, "iters": cfg.iters,
        "n_admitted": len(archive.nodes) - 1, "final_node": best.id,
        "dev_ndcg": best.ndcg, "test_ndcg": best_t.ndcg, "gap": best.ndcg - best_t.ndcg,
        "test_shift_ndcg": best_t.probe_mean, "shift_drop": best_t.ndcg - best_t.probe_mean,
        "test_energy": best_t.energy,
        "seed_dev_ndcg": archive.nodes[0].ndcg, "seed_test_ndcg": seed_t.ndcg,
        "seed_shift_drop": seed_t.ndcg - seed_t.probe_mean,
        "lineage_energy_increase_rate": inc, "lineage_edges": n_edges,
        "config": asdict(cfg),
    }
