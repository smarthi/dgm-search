"""Run a candidate retriever in a subprocess with a timeout, memory cap and scrubbed env.

This is process isolation, NOT a security sandbox. If the mutator is an LLM, run the whole
experiment inside a container with no network and no mounted secrets.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

WORKER = str(Path(__file__).with_name("worker.py"))


@dataclass
class SandboxResult:
    run: dict[str, list[str]] | None
    error: str | None


def run_candidate(path: str, corpus: dict[str, str], queries: dict[str, str], k: int,
                  timeout: float = 120.0, mem_mb: int = 2048) -> SandboxResult:
    def limit():
        try:
            import resource
            lim = mem_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
        except Exception:
            pass

    path = os.path.abspath(path)  # the worker runs with cwd set to a temp dir
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
    with tempfile.TemporaryDirectory() as td:
        inp, out = os.path.join(td, "in.json"), os.path.join(td, "out.json")
        with open(inp, "w", encoding="utf-8") as f:
            json.dump({"corpus": corpus, "queries": queries, "k": k}, f)
        try:
            p = subprocess.run([sys.executable, "-I", WORKER, path, inp, out], cwd=td, env=env,
                               capture_output=True, text=True, timeout=timeout, preexec_fn=limit)
        except subprocess.TimeoutExpired:
            return SandboxResult(None, f"timeout>{timeout}s")
        if p.returncode != 0:
            return SandboxResult(None, (p.stderr or "nonzero exit")[-600:])
        try:
            with open(out, encoding="utf-8") as f:
                return SandboxResult(json.load(f), None)
        except Exception as e:  # noqa: BLE001
            return SandboxResult(None, f"bad output: {e}")
