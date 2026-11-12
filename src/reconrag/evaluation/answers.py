"""Tier 2: end-to-end answers of two generators under seven conditions (test split, LLM judge labels).

* Accuracy over all questions and stale rate over current-rule questions (changed + deleted), with 95% bootstrap
  confidence intervals (2,000 replicates, resampling regulation families).
* Exact McNemar tests of ReconRAG against the three prompt-based baselines (Naive RAG, Recency prompt,
  Query-time judge), Holm-corrected within each generator and metric. Latest doc is not in the family because its
  low stale rate comes from discarding the base regulation.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict

from . import DATA

CONDITIONS = ["Naive_RAG", "Recency_Prompt", "Latest_Doc", "Query_Time_Judge", "ReconRAG", "ReconRAG_Route", "Oracle"]
BASELINES = ["Naive_RAG", "Recency_Prompt", "Query_Time_Judge"]
GENERATORS = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
TYPES = ["changed", "unchanged", "historical", "deleted"]
family = lambda r: r["question_id"].split("#")[0]


def boot_ci(items: list, stat, n: int = 2000, seed: int = 2026) -> tuple[float, float]:
    by = defaultdict(list)
    for x in items:
        by[family(x)].append(x)
    fams = sorted(by)
    rng = random.Random(seed)
    vals = sorted(stat([x for f in (rng.choice(fams) for _ in fams) for x in by[f]]) for _ in range(n))
    return vals[int(0.025 * n)], vals[int(0.975 * n)]


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def holm(pvals: list[float]) -> list[float]:
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, pvals[i] * (len(pvals) - rank)))
        adj[i] = running
    return adj


def run() -> tuple[list[dict], list[dict]]:
    rows = [json.loads(l) for l in open(DATA / "model_answers.jsonl", encoding="utf-8") if l.strip()]
    answers, tests = [], []
    for model in GENERATORS:
        m_rows = [r for r in rows if r["split"] == "test" and r["generator_model"] == model and r["llm_judge_label"]]
        complete = {q for q, c in Counter(r["question_id"] for r in m_rows).items() if c == len(CONDITIONS)}
        m_rows = [r for r in m_rows if r["question_id"] in complete]
        by = {c: {r["question_id"]: r for r in m_rows if r["condition"] == c} for c in CONDITIONS}
        correct = lambda s: sum(x["llm_judge_label"] == "correct" for x in s) / len(s)
        stale = lambda s: sum(x["llm_judge_label"] == "stale" for x in s) / len(s)
        for c in CONDITIONS:
            sub = list(by[c].values())
            cur = [r for r in sub if r["question_type"] in ("changed", "deleted")]
            a_lo, a_hi = boot_ci(sub, correct)
            s_lo, s_hi = boot_ci(cur, stale)
            row = {"generator": model, "condition": c, "n": len(sub),
                   "accuracy": round(correct(sub), 3), "accuracy_low": round(a_lo, 3), "accuracy_high": round(a_hi, 3),
                   "stale": round(stale(cur), 3), "stale_low": round(s_lo, 3), "stale_high": round(s_hi, 3)}
            for t in TYPES:
                row[t] = round(correct([r for r in sub if r["question_type"] == t]), 3)
            row["tokens"] = round(sum(r["total_tokens"] for r in sub) / len(sub))
            answers.append(row)

        cur_q = sorted(q for q, r in by["ReconRAG"].items() if r["question_type"] in ("changed", "deleted"))
        for metric, label, qids, better_is in (("stale", "stale", cur_q, "lower"),
                                               ("accuracy", "correct", sorted(by["ReconRAG"]), "higher")):
            raw = []
            for base in BASELINES:
                hit = lambda c, q: by[c][q]["llm_judge_label"] == label
                b_only = sum(hit(base, q) and not hit("ReconRAG", q) for q in qids)
                r_only = sum(hit("ReconRAG", q) and not hit(base, q) for q in qids)
                raw.append((base, b_only, r_only, mcnemar_exact(b_only, r_only)))
            for (base, b_only, r_only, p), p_adj in zip(raw, holm([x[3] for x in raw])):
                tests.append({"generator": model, "metric": metric, "baseline": base, "questions": len(qids),
                              "baseline_only": b_only, "reconrag_only": r_only,
                              "p": round(p, 4), "p_holm": round(p_adj, 4)})
    return answers, tests
