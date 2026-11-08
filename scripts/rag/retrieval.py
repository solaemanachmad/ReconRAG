"""Tier 2 Multi-Backend Retrieval Evaluation.

Evaluates 4 retrieval backends (TF-IDF, BM25, Dense MiniLM, Hybrid RRF) across
the test split of ``data/questions.jsonl`` (k=5 retrieved provisions).

Usage:
    python scripts/rag/retrieval.py            # Fast lexical (TF-IDF + BM25)
    python scripts/rag/retrieval.py --dense    # Full 4-backend incl. Dense MiniLM & Hybrid RRF
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from reconrag.parser.tfidf import tokens  # noqa: E402
from reconrag.rag import CONDITION_LABELS, ReconRAGBench, Retriever  # noqa: E402

DATA_DIR = ROOT / "data"


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
        texts = [f"{c['title']} Pasal {c['article']}: {c['text'][:600]}" for c in chunks]
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
        d_hits = self.dense.search(q, k=len(self.chunks), allow=allow)
        b_hits = self.bm25.search(q, k=len(self.chunks), allow=allow)
        rrf: dict[int, float] = defaultdict(float)
        for rank, (_, idx) in enumerate(d_hits, start=1):
            rrf[idx] += 1.0 / (self.k_rrf + rank)
        for rank, (_, idx) in enumerate(b_hits, start=1):
            rrf[idx] += 1.0 / (self.k_rrf + rank)
        return sorted(((score, idx) for idx, score in rrf.items()), reverse=True)[:k]


def main(with_dense: bool = False) -> dict:
    print("=" * 88)
    print("2. TIER 2: MULTI-BACKEND RETRIEVAL BENCHMARK (test split, k=5)")
    print("=" * 88)
    norm = lambda t: re.sub(r"\s+", " ", re.sub(r"[^\w\s%/.,()-]", " ", t.lower())).strip()
    qa = [
        json.loads(line)
        for line in open(DATA_DIR / "questions.jsonl", encoding="utf-8")
        if line.strip() and json.loads(line)["split"] == "test"
    ]
    bench_base = ReconRAGBench(k=5)
    chunks = bench_base.chunks

    backends: list[tuple[str, object]] = [
        ("TF-IDF", Retriever(chunks)),
        ("BM25", BM25Retriever(chunks)),
    ]
    if with_dense:
        ret_dense = DenseRetriever(chunks)
        backends.append(("Dense (MiniLM)", ret_dense))
        backends.append(("Hybrid (BM25+Dense)", HybridRetriever(ret_dense, backends[1][1])))  # type: ignore[arg-type]

    cond_list = ["Recency_Prompt", "Latest_Doc", "ReconRAG_StatusOnly", "ReconRAG", "ReconRAG_Route", "Oracle"]
    print(
        f"{'Backend':<20} | {'Condition':<28} | {'Changed↑':>8} | {'Stale↓':>7} | "
        f"{'Unch.↑':>7} | {'Hist.↑':>7} | {'DelStale↓':>9}"
    )
    print("-" * 88)
    results = {}
    for b_name, ret in backends:
        bench = ReconRAGBench(k=5, retriever=ret)
        results[b_name] = {}
        for c in cond_list:
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
                    m["del_stale"].append(any(jl in t for t in cur))
            avg = lambda k: sum(m[k]) / len(m[k])
            row_metrics = {
                "changed_in_force": avg("chg"),
                "changed_stale": avg("stale"),
                "unchanged_kept": avg("unch"),
                "historical_recalled": avg("hist"),
                "deleted_stale": avg("del_stale"),
            }
            results[b_name][c] = row_metrics
            print(
                f"{b_name:<20} | {CONDITION_LABELS[c]:<28} | {row_metrics['changed_in_force']:8.3f} | "
                f"{row_metrics['changed_stale']:7.3f} | {row_metrics['unchanged_kept']:7.3f} | "
                f"{row_metrics['historical_recalled']:7.3f} | {row_metrics['deleted_stale']:9.3f}"
            )
        print("-" * 88)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", action="store_true", help="Include Dense (MiniLM) and Hybrid RRF backends.")
    args = ap.parse_args()
    main(with_dense=args.dense)
