"""Standalone subprocess entry point. stdlib only; run as `python -I worker.py cand.py in.json out.json`.

Receives only corpus text and query text. Never sees relevance labels, so a candidate
cannot read them. Metrics are computed by the parent process.
"""
import importlib.util
import json
import sys


def main() -> None:
    cand, inp, out = sys.argv[1:4]
    with open(inp, encoding="utf-8") as f:
        data = json.load(f)
    spec = importlib.util.spec_from_file_location("candidate", cand)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    retriever = mod.Retriever(data["corpus"])
    k = data["k"]
    run = {}
    for qid, q in data["queries"].items():
        res = retriever.search(q, k)
        run[qid] = [str(d) for d in list(res)[:k]]
    with open(out, "w", encoding="utf-8") as f:
        json.dump(run, f)


if __name__ == "__main__":
    main()
