from __future__ import annotations

import ast
import math
import pprint
import random
import re

from .data import Dataset
from .evaluate import EvalResult

BEGIN, END = "# --- CONFIG BEGIN", "# --- CONFIG END"
_CFG_RE = re.compile(re.escape(BEGIN) + r"\n(.*?)\n" + re.escape(END), re.S)

FLOAT_RANGE = {"k1": (0.2, 3.0), "b": (0.0, 1.0), "bigram_weight": (0.0, 1.5), "prf_weight": (0.0, 1.5)}
INT_RANGE = {"prf_docs": (1, 10), "prf_terms": (1, 20)}
BOOLS = ("stopwords", "stem", "bigrams", "prf")


def read_config(code: str):
    m = _CFG_RE.search(code)
    if not m:
        return None, None
    return ast.literal_eval(m.group(1).split("=", 1)[1].strip()), m


def write_config(code: str, cfg: dict) -> str:
    _, m = read_config(code)
    block = BEGIN + "\nCONFIG = " + pprint.pformat(cfg, sort_dicts=False) + "\n" + END
    return code[: m.start()] + block + code[m.end():]


class RuleMutator:
    """Offline mutator: perturbs the CONFIG block. Exists to test the loop and gates without an LLM.
    It cannot produce new algorithms, so results with it are NOT evidence for the thesis."""

    def __init__(self, rng: random.Random):
        self.rng = rng

    def mutate(self, parent_code: str, parent_eval: EvalResult) -> str | None:
        cfg, _ = read_config(parent_code)
        if cfg is None:
            return None
        cfg = dict(cfg)
        keys = [k for k in list(FLOAT_RANGE) + list(INT_RANGE) + list(BOOLS) if k in cfg]
        for key in self.rng.sample(keys, k=self.rng.choice([1, 2])):
            if key in BOOLS:
                cfg[key] = not cfg[key]
            elif key in INT_RANGE:
                lo, hi = INT_RANGE[key]
                cfg[key] = min(hi, max(lo, cfg[key] + self.rng.choice([-2, -1, 1, 2])))
            else:
                lo, hi = FLOAT_RANGE[key]
                if key == "b":
                    v = cfg[key] + self.rng.gauss(0, 0.15)
                else:
                    v = cfg[key] * math.exp(self.rng.gauss(0, 0.35))
                cfg[key] = round(min(hi, max(lo, v)), 4)
        return write_config(parent_code, cfg)


_PROMPT = """You are improving a first-stage text retriever written in Python.

Contract (must hold): the file defines `class Retriever` with
    __init__(self, corpus: dict[str, str])      # doc_id -> text; build your index here
    search(self, query: str, k: int) -> list[str]   # doc ids, best first, only ids from the corpus
Rules: Python standard library and numpy only. Deterministic. No network, no file or environment access.
Do not hard-code query text, document ids, or anything memorised from the examples below; the code must
generalise to unseen queries on the same corpus. Keep indexing and search fast (a few minutes max).

Current retriever (dev nDCG@10 = {ndcg:.4f}):
```python
{code}
```

Some dev queries it handles worst (query -> gold document excerpt):
{failures}

Make ONE focused change that you expect to improve retrieval quality. Return the complete new file in a single
```python block, preceded by one line starting with `CHANGE:` describing the change."""


def extract_code(text: str) -> str | None:
    m = re.search(r"```python\n(.*?)```", text, re.S)
    if not m:
        return None
    code = m.group(1)
    try:
        ast.parse(code)
    except SyntaxError:
        return None
    return code if "class Retriever" in code else None


class LLMMutator:
    """Asks an LLM for a rewritten retriever. The prompt is identical across arms and blind to the gate."""

    def __init__(self, dev: Dataset, model: str, max_tokens: int = 12000):
        import anthropic  # optional dependency: pip install 'dgm-search[llm]'

        self.client = anthropic.Anthropic()
        self.dev, self.model, self.max_tokens = dev, model, max_tokens

    def mutate(self, parent_code: str, parent_eval: EvalResult) -> str | None:
        lines = []
        for qid, _ in parent_eval.worst:
            gold = next(iter(self.dev.qrels[qid]))
            lines.append(f"- {self.dev.queries[qid]!r} -> {self.dev.corpus[gold][:300]!r}")
        prompt = _PROMPT.format(ndcg=parent_eval.ndcg, code=parent_code, failures="\n".join(lines))
        msg = self.client.messages.create(model=self.model, max_tokens=self.max_tokens,
                                          messages=[{"role": "user", "content": prompt}])
        text = "".join(b.text for b in msg.content if b.type == "text")
        return extract_code(text)


def make_mutator(kind: str, rng: random.Random, dev: Dataset, model: str):
    if kind == "rule":
        return RuleMutator(rng)
    if kind == "llm":
        return LLMMutator(dev, model)
    raise ValueError(kind)
