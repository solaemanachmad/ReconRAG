"""Human Validation of the LLM Answer Judge (Section 5.4).

Computes agreement and Cohen's kappa on the 53 stratified model answers
manually verified against statutory text in ``data/human_judge_validation.csv``.

Usage:
    python scripts/rag/validate_judge.py
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def cohens_kappa(y_true: list[str], y_pred: list[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    n = len(y_true)
    if n == 0:
        return 0.0
    p_o = sum(1 for a, b in zip(y_true, y_pred) if a == b) / n
    p_e = sum((sum(1 for a in y_true if a == c) / n) * (sum(1 for b in y_pred if b == c) / n) for c in labels)
    return 1.0 if p_e >= 1.0 else (p_o - p_e) / (1.0 - p_e)


def main() -> dict:
    path = ROOT / "data" / "human_judge_validation.csv"
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    human = [r["human_label"] for r in rows]
    llm = [r["llm_judge_label"] for r in rows]
    rule = [r["token_rule_label"] for r in rows]

    n = len(rows)
    acc_llm = sum(1 for h, p in zip(human, llm) if h == p) / n
    kap_llm = cohens_kappa(human, llm)
    acc_rule = sum(1 for h, p in zip(human, rule) if h == p) / n
    kap_rule = cohens_kappa(human, rule)

    print("=" * 88)
    print(f"4. HUMAN JUDGE VALIDATION ({n} Stratified Model Outputs, Section 5.4)")
    print("=" * 88)
    print(f"LLM Judge vs. Human Gold   : Agreement = {acc_llm:.3f} ({sum(h == p for h, p in zip(human, llm))}/{n}), Cohen's kappa = {kap_llm:.3f}")
    print(f"Token-Rule vs. Human Gold  : Agreement = {acc_rule:.3f} ({sum(h == p for h, p in zip(human, rule))}/{n}), Cohen's kappa = {kap_rule:.3f}")
    return {
        "n": n,
        "llm_accuracy": acc_llm,
        "llm_kappa": kap_llm,
        "rule_accuracy": acc_rule,
        "rule_kappa": kap_rule,
    }


if __name__ == "__main__":
    main()
