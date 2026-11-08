"""Tier 2 End-to-End Answer Evaluation, Bootstrap CIs & McNemar Tests.

Evaluates all generated answers in ``data/model_answers.jsonl`` across the two
generator backbones (openai/gpt-oss-120b and qwen/qwen3.8-27b) and 7 RAG conditions:
  - 2,000-replicate family-cluster bootstrap 95% confidence intervals
  - Exact binomial McNemar tests with Holm-Bonferroni step-down correction

Usage:
    python scripts/rag/evaluate_qa.py
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

COND_ORDER = [
    ("Naive_RAG", "Naive RAG"),
    ("Recency_Prompt", "Recency prompt"),
    ("Latest_Doc", "Latest doc"),
    ("Query_Time_Judge", "Query-time judge"),
    ("ReconRAG", "ReconRAG (Ours)"),
    ("ReconRAG_Route", "+ query routing (ablation)"),
    ("Oracle", "Oracle ceiling"),
]


def boot_ci(items: list, fam_of, stat, n: int = 2000, seed: int = 2026) -> tuple[float, float]:
    fams = sorted({fam_of(x) for x in items})
    by = defaultdict(list)
    for x in items:
        by[fam_of(x)].append(x)
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        sample = [x for f in (rng.choice(fams) for _ in fams) for x in by[f]]
        vals.append(stat(sample))
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n)]


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * p)


def main() -> dict:
    print("=" * 98)
    print("3. TIER 2: END-TO-END RAG ANSWERS & STATISTICAL TESTS")
    print("=" * 98)
    rows = [
        json.loads(line)
        for line in open(DATA_DIR / "model_answers.jsonl", encoding="utf-8")
        if line.strip()
    ]

    summary = {}
    for model in ("openai/gpt-oss-120b", "qwen/qwen3.8-27b"):
        m_rows = [
            r for r in rows
            if r["split"] == "test" and r["generator_model"] == model and r["llm_judge_label"] is not None
        ]
        q_counts = Counter(r["question_id"] for r in m_rows)
        valid_qids = {qid for qid, cnt in q_counts.items() if cnt == len(COND_ORDER)}
        m_rows = [r for r in m_rows if r["question_id"] in valid_qids]

        print(f"\nGenerator: {model} (n={len(valid_qids)} paired test questions, LLM Judge)")
        print(
            f"{'Condition':<28} | {'Accuracy [95% CI]':<21} | {'Stale [95% CI]':<21} | "
            f"{'Chg':>5} {'Unch':>5} {'Hist':>5} | {'Tokens':>6}"
        )
        print("-" * 98)

        by_cond_qid = {}
        summary[model] = {}
        for c_code, c_label in COND_ORDER:
            sub = [r for r in m_rows if r["condition"] == c_code]
            by_cond_qid[c_code] = {r["question_id"]: r for r in sub}
            acc = sum(r["llm_judge_label"] == "correct" for r in sub) / len(sub)
            acc_lo, acc_hi = boot_ci(
                sub,
                lambda x: x["question_id"].split("#")[0],
                lambda s: sum(x["llm_judge_label"] == "correct" for x in s) / len(s),
            )
            cur = [r for r in sub if r["question_type"] in ("changed", "deleted")]
            stale = sum(r["llm_judge_label"] == "stale" for r in cur) / len(cur)
            st_lo, st_hi = boot_ci(
                cur,
                lambda x: x["question_id"].split("#")[0],
                lambda s: sum(x["llm_judge_label"] == "stale" for x in s) / len(s),
            )
            by_t = {}
            for qt in ("changed", "unchanged", "historical"):
                qt_sub = [r for r in sub if r["question_type"] == qt]
                by_t[qt] = sum(r["llm_judge_label"] == "correct" for r in qt_sub) / len(qt_sub)
            toks = round(sum(r["total_tokens"] for r in sub) / len(sub))
            summary[model][c_code] = {
                "acc": acc,
                "acc_ci": (acc_lo, acc_hi),
                "stale": stale,
                "stale_ci": (st_lo, st_hi),
                "by_type": by_t,
                "tokens": toks,
            }
            print(
                f"{c_label:<28} | {acc:.2f} [{acc_lo:.2f}–{acc_hi:.2f}]      | "
                f"{stale:.2f} [{st_lo:.2f}–{st_hi:.2f}]      | "
                f"{by_t['changed']:5.2f} {by_t['unchanged']:5.2f} {by_t['historical']:5.2f} | {toks:6d}"
            )

        cur_qids = sorted(
            qid for qid, r in by_cond_qid["ReconRAG"].items() if r["question_type"] in ("changed", "deleted")
        )
        p_raw = []
        for base_code in ("Naive_RAG", "Recency_Prompt", "Query_Time_Judge"):
            b_only = sum(
                by_cond_qid[base_code][q]["llm_judge_label"] == "stale"
                and by_cond_qid["ReconRAG"][q]["llm_judge_label"] != "stale"
                for q in cur_qids
            )
            r_only = sum(
                by_cond_qid[base_code][q]["llm_judge_label"] != "stale"
                and by_cond_qid["ReconRAG"][q]["llm_judge_label"] == "stale"
                for q in cur_qids
            )
            p_raw.append((mcnemar_exact(b_only, r_only), base_code, b_only, r_only))
        p_raw.sort()
        print("  McNemar Stale Reduction (ReconRAG vs. Baselines, Holm-Bonferroni corrected):")
        running_max = 0.0
        for idx, (p_val, base_code, b_only, r_only) in enumerate(p_raw):
            # Holm step-down: adjusted p-values must be monotonically non-decreasing.
            running_max = max(running_max, min(1.0, p_val * (len(p_raw) - idx)))
            p_holm = running_max
            print(f"    vs. {base_code:<18}: discordant={b_only}:{r_only}, p={p_val:.4f}, p_Holm={p_holm:.4f}")
    return summary


if __name__ == "__main__":
    main()
