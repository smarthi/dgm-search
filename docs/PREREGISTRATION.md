# Pre-registration v1

Commit this file, tag the commit `prereg-v1`, and only then run any experiment whose result you intend to
report. Changes after that go in the deviations log at the bottom, with a date and a reason.

## Question

In Darwin-Godel-style open-ended self-improvement of a retrieval pipeline, does admitting a variant to the archive
only if a stability energy V is non-increasing (a Lyapunov-style descent condition) produce final agents that
generalize better than admitting on dev score alone?

## Manipulation

Four arms, identical in parent selection, mutator, mutator prompt, iteration budget and final-selection rule.
Only the admission gate (`src/dgmsearch/gates.py`) differs.

| arm | admission rule |
|---|---|
| `open` | valid and dev nDCG >= 0.9 x seed (DGM-like open archive) |
| `score` | dev nDCG >= parent - 0.005 |
| `score_stab` | `score` and V_child <= V_parent + 0.005 |
| `score_size` | `score` and lines(child) <= 1.10 x lines(parent) (placebo regularizer) |

The mutator is blind to the gate: its prompt shows parent code, parent dev nDCG and worst dev queries, never V.

## Stability energy

Phase 1 uses the operational proxy in `stability.py`: degradation of nDCG@10 under typo / word-drop / shuffle
perturbations of dev queries (two seeds, intensity 0.25). It is **not** the Lyapunov-RAG formalization. Phase 2
replaces it (same signature) and re-runs; a phase-1 result says nothing about the formalization.

## Data and splits

- Selection uses `dev` only. The loop never loads `test`.
- `test` is touched once per run, after the final agent is chosen (`argmax` dev nDCG among admitted nodes, for every arm).
- Shift evaluation perturbs `test` queries with different seeds and higher intensity (0.5) than the in-loop probes,
  so the gate cannot have trained on the exact perturbations.
- Datasets: BEIR SciFact first (small); one more BEIR set before any claim of generality. Toy data is for CI only.

## Hypotheses (final agent, per run)

- H1: `test_ndcg` is higher for `score_stab` than `score`.
- H2: `gap` (dev - test nDCG) is smaller for `score_stab` than `score`.
- H3: `shift_drop` (test nDCG - shifted-test nDCG) is smaller for `score_stab` than `score`.
- H4 (mechanism check): `lineage_energy_increase_rate` is higher for `score` than `score_stab`, i.e. score-only
  admission accumulates fragility along lineages.

`score_size` is the control for the reading "any extra constraint helps". If `score_stab` beats `score` but not
`score_size`, the stability claim is not supported.

## Analysis

Primary: difference of means (`score_stab` - `score`) on `test_ndcg`, `gap`, `shift_drop`, with 95% bootstrap CIs
over seeds (`dgmsearch analyze`). Minimum 10 seeds per arm for any reported claim; 5 for pilots, labelled as pilots.
Report all four arms and the seed BM25 baseline, including arms that look bad. No outcome-dependent seed dropping.
Runs that crash for infrastructure reasons are re-run with the same seed; mutator failures count as spent iterations.

## What each outcome means (decided before looking)

- CI of the `test_ndcg` and `shift_drop` differences both exclude 0 in the predicted direction, and `score_size` does not
  match: supports the thesis for this proxy, dataset and mutator.
- Intervals include 0: report as a null result. Do not retune tolerances to rescue it.
- `score_stab` improves `shift_drop` but lowers `test_ndcg`: report as a robustness/accuracy tradeoff, not a win.

## Known threats

- LLM mutators are nondeterministic; seeds fix selection, not generations. Log model id and date.
- The base model may have seen BEIR data. Prefer a second corpus and report SciFact-specific caveats.
- Proxy energy may correlate with something trivial (e.g. fewer features). `score_size` partially controls this;
  inspect the admitted lineages by hand.
- Rule-mutator results exercise the machinery only. They are not evidence.
- Candidate code runs in a subprocess, not a security sandbox. Use a container for LLM runs.

## Deviations log

(none yet)
