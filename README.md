# dgm-search

A falsifiable test of one claim: **Darwin-Godel-style self-improvement of a retrieval pipeline overfits its dev
set, and a stability gate on archive admission reduces that.**

The loop follows the Darwin Godel Machine pattern (Sakana AI): keep an archive of agents, sample a parent
(favouring strong, under-explored nodes), have a mutator rewrite its code, evaluate the child, and decide whether it
enters the archive. Here the agent is a retriever (`Retriever(corpus).search(query, k)`), the mutator is an LLM
(or an offline rule mutator for testing), and the only thing varied between experimental arms is the **admission gate**.

```
seed BM25 ─► archive ─► select parent ─► mutate code ─► sandbox eval (dev nDCG, stability energy V)
                ▲                                                     │
                └──────────────── admit? (arm-specific gate) ◄────────┘
final: argmax dev nDCG in archive ─► held-out test nDCG, shifted-test drop, dev→test gap (once per run)
```

| arm | gate |
|---|---|
| `open` | any valid child not far below the seed |
| `score` | dev nDCG not worse than parent |
| `score_stab` | `score` and V non-increasing (Lyapunov-style descent) |
| `score_size` | `score` and code size bounded (placebo regularizer) |

Read [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md) first. It fixes hypotheses, analysis and what each outcome means
before any result exists.

## Status: what is and is not real

- Real and tested: the loop, archive, sandboxed evaluation, gates, held-out final eval, bootstrap analysis (`pytest`).
- **Placeholder:** `stability.py` is an operational proxy (nDCG degradation under query perturbation). Swap in the
  Lyapunov-RAG energy there (same signature) to test the formalization itself.
- **Not yet built:** multi-hop. The retriever interface is single-shot `search(query, k)`, so phase 1 tests the gate on
  single-hop retrieval. A multi-hop variant needs an iterative interface and a chain-level energy.
- **Untested here:** `scripts/prepare_beir.py` (no network route in the authoring sandbox) and the LLM mutator
  (no API key). Run the former once and eyeball the output; smoke-test the latter with `--iters 3`.
- The rule mutator only perturbs BM25 hyperparameters and flags. Runs with it validate plumbing. They are not evidence.

## Quick start

```bash
pip install -e '.[dev]'          # add ',llm' for the Anthropic mutator
pytest                           # offline, a few seconds
dgmsearch toy-data --out data/toy
dgmsearch sweep --data data/toy --mutator rule --iters 30 --seeds 0-4
dgmsearch analyze --runs runs

# real experiment
python scripts/prepare_beir.py scifact data/scifact
export ANTHROPIC_API_KEY=...     # run inside a container: candidate code is LLM-written
dgmsearch sweep --data data/scifact --mutator llm --iters 40 --seeds 0-9
dgmsearch analyze --runs runs --mutator llm
```

Cost: arms x seeds x iters mutator calls (4 x 10 x 40 = 1,600) plus evaluation CPU. Pilot with 2 seeds and 15 iterations first.

## Layout

`src/dgmsearch/` `loop.py` (evolve + final eval), `gates.py` (the manipulation), `stability.py` (energy),
`mutators.py`, `evaluate.py`, `sandbox.py`/`worker.py` (subprocess isolation; candidate sees corpus and query text only),
`archive.py`, `analysis.py`. Every run writes `archive.jsonl`, `log.jsonl`, admitted agents under `agents/`, rejected
ones under `rejected/` (useful for reading what the gate filtered), and `final.json`.

## Writing it up

Report the null if it is one. The post is worth writing either way: "stability gating did / did not help, here is the
pre-registered test" is a stronger piece than a thesis with no evidence behind it.
