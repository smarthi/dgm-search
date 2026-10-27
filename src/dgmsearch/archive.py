from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from .evaluate import EvalResult


@dataclass
class Node:
    id: int
    parent: int | None
    path: str
    ndcg: float
    energy: float
    n_lines: int
    it: int
    children: int = 0


class Archive:
    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir)
        (self.dir / "agents").mkdir(parents=True, exist_ok=True)
        self.nodes: list[Node] = []
        self.evals: dict[int, EvalResult] = {}

    def add(self, code: str, parent: Node | None, ev: EvalResult, it: int) -> Node:
        nid = len(self.nodes)
        path = self.dir / "agents" / f"{nid}.py"
        path.write_text(code, encoding="utf-8")
        node = Node(nid, parent.id if parent else None, str(path), ev.ndcg, ev.energy, ev.n_lines, it)
        if parent:
            parent.children += 1
        self.nodes.append(node)
        self.evals[nid] = ev
        with open(self.dir / "archive.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(node)) + "\n")
        return node

    def select_parent(self, rng: random.Random) -> Node:
        """DGM-style: favour high performers, discount nodes that already have many children."""
        med = statistics.median(n.ndcg for n in self.nodes)
        w = [(1 / (1 + math.exp(-10 * (n.ndcg - med)))) / (1 + n.children) for n in self.nodes]
        return rng.choices(self.nodes, weights=w)[0]

    def best(self) -> Node:
        return max(self.nodes, key=lambda n: (n.ndcg, -n.id))

    def lineage_energy_increase_rate(self) -> tuple[float, int]:
        """Fraction of admitted parent->child edges where V went up. Measures drift the gate let through."""
        edges = [(self.nodes[n.parent], n) for n in self.nodes if n.parent is not None]
        if not edges:
            return 0.0, 0
        up = sum(1 for p, c in edges if c.energy > p.energy + 1e-12)
        return up / len(edges), len(edges)
