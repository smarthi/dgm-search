from __future__ import annotations

import argparse
import json

from .analysis import analyze
from .config import ARMS, RunConfig
from .data import make_toy
from .loop import run_arm


def _seeds(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="dgmsearch")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("toy-data", help="generate the synthetic smoke-test dataset")
    t.add_argument("--out", default="data/toy")

    r = sub.add_parser("run", help="run one arm, one seed")
    s = sub.add_parser("sweep", help="run arms x seeds")
    for p in (r, s):
        p.add_argument("--data", default="data/toy")
        p.add_argument("--out", default="runs")
        p.add_argument("--mutator", choices=["rule", "llm"], default="rule")
        p.add_argument("--iters", type=int, default=40)
        p.add_argument("--model", default="claude-sonnet-5-5")
        p.add_argument("--timeout", type=float, default=120.0)
    r.add_argument("--arm", choices=ARMS, required=True)
    r.add_argument("--seed", type=int, default=0)
    s.add_argument("--arms", default=",".join(ARMS))
    s.add_argument("--seeds", default="0-4")

    a = sub.add_parser("analyze", help="summarise runs/")
    a.add_argument("--runs", default="runs")
    a.add_argument("--mutator", default=None)

    args = ap.parse_args(argv)
    if args.cmd == "toy-data":
        make_toy(args.out)
        print(f"wrote {args.out}")
    elif args.cmd == "run":
        cfg = RunConfig(data_dir=args.data, out_dir=args.out, arm=args.arm, mutator=args.mutator,
                        iters=args.iters, seed=args.seed, model=args.model, timeout=args.timeout)
        res = run_arm(cfg)
        print(json.dumps({k: v for k, v in res.items() if k != "config"}, indent=2))
    elif args.cmd == "sweep":
        for seed in _seeds(args.seeds):
            for arm in args.arms.split(","):
                cfg = RunConfig(data_dir=args.data, out_dir=args.out, arm=arm, mutator=args.mutator,
                                iters=args.iters, seed=seed, model=args.model, timeout=args.timeout)
                res = run_arm(cfg)
                print(f"{arm:11s} seed={seed} dev={res['dev_ndcg']:.3f} test={res['test_ndcg']:.3f} "
                      f"shift_drop={res['shift_drop']:.3f} admitted={res['n_admitted']}", flush=True)
    elif args.cmd == "analyze":
        print(analyze(args.runs, args.mutator))


if __name__ == "__main__":
    main()
