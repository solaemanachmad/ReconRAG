"""Tier 2: what the generator would see. Retrieval with k=5 on the test questions, without any LLM.

Backends: TF-IDF and BM25 always; Dense (multilingual MiniLM) and hybrid RRF with ``dense=True``
(needs ``pip install -e ".[dense]"``).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

import numpy as np

from ..parser.tfidf import tokens
from ..rag import ReconRAGBench, Retriever
from . import DATA as DATA_DIR


class BM25Retriever:
    def __init__(self, chunks: list[dict]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.corpus = [tokens(c["title"] + " " + c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus)

    def search(self, q: str, k: int = 5, allow=None) -> list[tuple[float, int]]:
        scores = self.bm25.get_scores(tokens(q))
        sc = []
        for i, s in enumerate(scores):
            if allow is not None and not allow(self.chunks[i]):
                continue
            if s > 0:
                sc.append((float(s), i))
        return sorted(sc, reverse=True)[:k]


class DenseRetriever:
    def __init__(self, chunks: list[dict], model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        texts = [f"{c['title']} - Pasal {c['article']}: {c['text']}" for c in chunks]  # as in the paper's run
        self.emb = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64)

    def search(self, q: str, k: int = 5, allow=None) -> list[tuple[float, int]]:
        qv = self.model.encode([q], normalize_embeddings=True, show_progress_bar=False)[0]
        sims = np.matmul(self.emb, qv)
        sc = []
        for i, s in enumerate(sims):
            if allow is not None and not allow(self.chunks[i]):
                continue
            sc.append((float(s), i))
        return sorted(sc, reverse=True)[:k]


class HybridRetriever:
    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, k_rrf: int = 60):
        self.chunks = dense.chunks
        self.dense = dense
        self.bm25 = bm25
        self.k_rrf = k_rrf

    def search(self, q: str, k: int = 5, allow=None) -> list[tuple[float, int]]:
        # fuse the top 4k candidates of each retriever (as in the paper's run); ties keep insertion order
        d_hits = self.dense.search(q, k=4 * k, allow=allow)
        b_hits = self.bm25.search(q, k=4 * k, allow=allow)
        rrf: dict[int, float] = defaultdict(float)
        for rank, (_, idx) in enumerate(d_hits, start=1):
            rrf[idx] += 1.0 / (self.k_rrf + rank)
        for rank, (_, idx) in enumerate(b_hits, start=1):
            rrf[idx] += 1.0 / (self.k_rrf + rank)
        return [(score, idx) for idx, score in sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:k]]


CONDITIONS = ["Recency_Prompt", "Latest_Doc", "ReconRAG_StatusOnly", "ReconRAG", "ReconRAG_Route", "Oracle"]


def run(dense: bool = False) -> list[dict]:
    norm = lambda t: re.sub(r"\s+", " ", re.sub(r"[^\w\s%/.,()-]", " ", t.lower())).strip()
    qa = [q for q in map(json.loads, open(DATA_DIR / "questions.jsonl", encoding="utf-8")) if q["split"] == "test"]
    chunks = ReconRAGBench(k=5).chunks
    backends: list[tuple[str, object]] = [("TF-IDF", Retriever(chunks)), ("BM25", BM25Retriever(chunks))]
    if dense:
        ret_dense = DenseRetriever(chunks)
        backends += [("Dense", ret_dense), ("Hybrid", HybridRetriever(ret_dense, backends[1][1]))]  # type: ignore[arg-type]
    rows = []
    for b_name, ret in backends:
        bench = ReconRAGBench(k=5, retriever=ret)
        for c in CONDITIONS:
            m = defaultdict(list)
            for q in qa:
                seen = [(norm(x["raw"]), x["status"]) for x in bench.context(q["question"], c)]
                jb, jl = norm(q["in_force_answer"]), norm(q["superseded_answer"])
                cur = [t for t, s in seen if s == "in_force"]
                qt = q["question_type"]
                if qt == "changed":
                    m["chg"].append(any(jb in t for t in cur))
                    m["stale"].append(any(jl in t and jb not in t for t in cur))
                elif qt == "unchanged":
                    m["unch"].append(any(jb in t for t in cur))
                elif qt == "historical":
                    m["hist"].append(any(jl in t for t, _ in seen))
                elif qt == "deleted":
                    m["del"].append(any(jl in t for t in cur))
            avg = lambda k: round(sum(m[k]) / len(m[k]), 3)
            rows.append({"backend": b_name, "condition": c, "changed_inforce": avg("chg"), "changed_stale": avg("stale"),
                         "unchanged": avg("unch"), "historical": avg("hist"), "deleted_stale": avg("del")})
    return rows
