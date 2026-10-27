"""Convert a BEIR dataset into this repo's format with a fixed dev/test split of the labelled queries.

    python scripts/prepare_beir.py scifact data/scifact

Downloads https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/<name>.zip (UNTESTED here:
the authoring sandbox had no route to that host; run it once and eyeball the output). BEIR ships labelled
queries in qrels/test.tsv (and train/dev for some sets); we pool whatever labelled queries exist and split
50/50 with a fixed seed, so the loop never sees test queries.
"""
import io
import json
import random
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"


def main(name: str, out: str, split_seed: int = 0) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL.format(name)).read()))
    base = name + "/"
    corpus = [json.loads(l) for l in zf.read(base + "corpus.jsonl").decode().splitlines() if l.strip()]
    queries = {}
    for l in zf.read(base + "queries.jsonl").decode().splitlines():
        if l.strip():
            q = json.loads(l)
            queries[q["_id"]] = q["text"]
    qrels: dict[str, dict[str, int]] = {}
    for member in zf.namelist():
        if member.startswith(base + "qrels/") and member.endswith(".tsv"):
            for line in zf.read(member).decode().splitlines()[1:]:
                parts = line.split("\t")
                if len(parts) == 3 and int(parts[2]) > 0:
                    qrels.setdefault(parts[0], {})[parts[1]] = int(parts[2])
    qids = sorted(q for q in qrels if q in queries)
    random.Random(split_seed).shuffle(qids)
    half = len(qids) // 2

    def write(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    write(out / "corpus.jsonl", ({"id": d["_id"], "text": (d.get("title", "") + " " + d["text"]).strip()} for d in corpus))
    for split, ids in (("dev", qids[:half]), ("test", qids[half:])):
        write(out / f"{split}.jsonl", ({"id": q, "text": queries[q], "relevant": qrels[q]} for q in ids))
    print(f"{name}: {len(corpus)} docs, {half} dev / {len(qids) - half} test queries -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
