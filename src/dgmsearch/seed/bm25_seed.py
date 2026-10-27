"""Seed retriever: configurable BM25. The evolve loop mutates this file.

Contract (do not change): class Retriever(corpus: dict[str, str]) with
    search(query: str, k: int) -> list[str]   # doc ids, best first
"""
import math
import re
from collections import defaultdict

# --- CONFIG BEGIN
CONFIG = {
    "k1": 1.2,
    "b": 0.75,
    "stopwords": False,
    "stem": False,
    "bigrams": False,
    "bigram_weight": 0.3,
    "prf": False,
    "prf_docs": 3,
    "prf_terms": 5,
    "prf_weight": 0.5,
}
# --- CONFIG END

_STOP = frozenset("a an and are as at be by for from has he in is it its of on that the to was were will with".split())
_TOK = re.compile(r"[a-z0-9]+")


def _stem(t):
    for suf in ("ing", "ed", "es", "s"):
        if len(t) > len(suf) + 3 and t.endswith(suf):
            return t[: -len(suf)]
    return t


class Retriever:
    def __init__(self, corpus):
        self.cfg = dict(CONFIG)
        self.ids = list(corpus.keys())
        self.postings = defaultdict(list)
        self.doc_len = []
        self.doc_terms = []
        for i, did in enumerate(self.ids):
            toks = self._analyze(corpus[did])
            self.doc_len.append(len(toks))
            counts = defaultdict(int)
            for t in toks:
                counts[t] += 1
            if self.cfg["bigrams"]:
                for a, b in zip(toks, toks[1:]):
                    counts[a + "_" + b] += 1
            for t, c in counts.items():
                self.postings[t].append((i, c))
            self.doc_terms.append(counts)
        self.N = len(self.ids)
        self.avg = (sum(self.doc_len) / self.N) if self.N else 1.0

    def _analyze(self, text):
        toks = _TOK.findall(text.lower())
        if self.cfg["stopwords"]:
            toks = [t for t in toks if t not in _STOP]
        if self.cfg["stem"]:
            toks = [_stem(t) for t in toks]
        return toks

    def _idf(self, t):
        df = len(self.postings.get(t, ()))
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def _score(self, qw):
        k1, b = self.cfg["k1"], self.cfg["b"]
        scores = defaultdict(float)
        for t, w in qw.items():
            post = self.postings.get(t)
            if not post:
                continue
            idf = self._idf(t)
            for i, tf in post:
                norm = k1 * (1 - b + b * self.doc_len[i] / self.avg)
                scores[i] += w * idf * tf * (k1 + 1) / (tf + norm)
        return scores

    def search(self, query, k):
        toks = self._analyze(query)
        qw = defaultdict(float)
        for t in toks:
            qw[t] += 1.0
        if self.cfg["bigrams"]:
            for a, bg in zip(toks, toks[1:]):
                qw[a + "_" + bg] += self.cfg["bigram_weight"]
        scores = self._score(qw)
        if self.cfg["prf"] and scores:
            top = sorted(scores, key=scores.get, reverse=True)[: self.cfg["prf_docs"]]
            fb = defaultdict(float)
            for i in top:
                for t, c in self.doc_terms[i].items():
                    if "_" in t:
                        continue
                    fb[t] += c * self._idf(t)
            for t, _ in sorted(fb.items(), key=lambda x: -x[1])[: self.cfg["prf_terms"]]:
                qw[t] += self.cfg["prf_weight"]
            scores = self._score(qw)
        ranked = sorted(scores, key=scores.get, reverse=True)[:k]
        return [self.ids[i] for i in ranked]
