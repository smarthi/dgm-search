import json
import random

import pytest

from dgmsearch.config import ARMS, RunConfig
from dgmsearch.data import load_dataset, make_toy
from dgmsearch.evaluate import EvalResult, evaluate
from dgmsearch.gates import admit
from dgmsearch.loop import run_arm, seed_code
from dgmsearch.mutators import RuleMutator, extract_code, read_config
from dgmsearch.perturb import build_probe_queries
from dgmsearch.stability import energy


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    root = tmp_path_factory.mktemp("toy")
    make_toy(root, n_docs=120, n_topics=8, n_dev=20, n_test=20)
    return root


def test_probes_are_deterministic():
    q = {"a": "alpha beta gamma delta"}
    assert build_probe_queries(q, (0,), 0.5) == build_probe_queries(q, (0,), 0.5)


def test_energy_zero_when_no_drop():
    assert energy(0.5, {"x": 0.6, "y": 0.5}) == 0.0
    assert energy(0.5, {"x": 0.2}) > 0.0


def test_seed_is_valid_and_better_than_chance(toy, tmp_path):
    p = tmp_path / "seed.py"
    p.write_text(seed_code())
    ev = evaluate(str(p), load_dataset(toy, "dev"), RunConfig(data_dir=str(toy)), (0,), 0.25)
    assert ev.valid and ev.ndcg > 0.3


def test_rule_mutation_changes_config_and_runs(toy, tmp_path):
    parent = seed_code()
    child = RuleMutator(random.Random(1)).mutate(parent, EvalResult(True))
    assert child != parent and read_config(child)[0] != read_config(parent)[0]
    p = tmp_path / "c.py"
    p.write_text(child)
    assert evaluate(str(p), load_dataset(toy, "dev"), RunConfig(data_dir=str(toy)), (0,), 0.25).valid


def test_broken_candidate_is_invalid(toy, tmp_path):
    p = tmp_path / "bad.py"
    p.write_text("class Retriever:\n    def __init__(self, c): pass\n    def search(self, q, k): return ['nope']\n")
    ev = evaluate(str(p), load_dataset(toy, "dev"), RunConfig(data_dir=str(toy)), (0,), 0.25)
    assert not ev.valid


def test_candidate_cannot_see_labels(toy, tmp_path):
    p = tmp_path / "peek.py"
    p.write_text("import os\nclass Retriever:\n    def __init__(self, c):\n        raise RuntimeError(str(sorted(os.environ)))\n    def search(self, q, k): return []\n")
    ev = evaluate(str(p), load_dataset(toy, "dev"), RunConfig(data_dir=str(toy)), (0,), 0.25)
    assert not ev.valid and "KEY" not in (ev.error or "")


def test_gates():
    seed = EvalResult(True, ndcg=0.5, energy=0.10, n_lines=100)
    parent = EvalResult(True, ndcg=0.6, energy=0.10, n_lines=100)
    better_fragile = EvalResult(True, ndcg=0.7, energy=0.30, n_lines=100)
    better_big = EvalResult(True, ndcg=0.7, energy=0.10, n_lines=200)
    worse = EvalResult(True, ndcg=0.4, energy=0.0, n_lines=100)
    c = lambda arm: RunConfig(arm=arm)  # noqa: E731
    assert admit(c("score"), better_fragile, parent, seed)[0]
    assert not admit(c("score_stab"), better_fragile, parent, seed)[0]
    assert admit(c("score_stab"), better_big, parent, seed)[0]
    assert not admit(c("score_size"), better_big, parent, seed)[0]
    assert not admit(c("score"), worse, parent, seed)[0]
    assert admit(c("open"), EvalResult(True, ndcg=0.46, n_lines=1), parent, seed)[0]
    assert not admit(c("open"), EvalResult(False), parent, seed)[0]


def test_extract_code():
    assert extract_code("x\n```python\nclass Retriever:\n    pass\n```") is not None
    assert extract_code("```python\ndef f(:\n```") is None
    assert extract_code("no code here") is None


@pytest.mark.parametrize("arm", ARMS)
def test_end_to_end_all_arms(toy, tmp_path, arm):
    cfg = RunConfig(data_dir=str(toy), out_dir=str(tmp_path), arm=arm, iters=6, seed=3)
    res = run_arm(cfg)
    assert res["dev_ndcg"] >= res["seed_dev_ndcg"] - 1e-9
    saved = json.loads((tmp_path / f"{arm}_rule_s3" / "final.json").read_text())
    assert saved["arm"] == arm
