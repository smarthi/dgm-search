import json

from dgmsearch.cli import main


def test_relative_paths_work(tmp_path, monkeypatch):
    """Regression: the sandbox worker runs in a temp cwd, so candidate paths must be absolute."""
    monkeypatch.chdir(tmp_path)
    main(["toy-data", "--out", "data/toy"])
    main(["run", "--data", "data/toy", "--out", "runs", "--arm", "score_stab", "--iters", "3", "--seed", "0"])
    assert json.loads((tmp_path / "runs/score_stab_rule_s0/final.json").read_text())["arm"] == "score_stab"
