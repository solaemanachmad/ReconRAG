"""Validation of the answer graders against one human rater on 53 stratified answers (``data/judge_validation.csv``)."""

from __future__ import annotations

import csv

from . import DATA


def cohens_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    labels = set(a) | set(b)
    p_o = sum(x == y for x, y in zip(a, b)) / n
    p_e = sum((a.count(c) / n) * (b.count(c) / n) for c in labels)
    return 1.0 if p_e >= 1.0 else (p_o - p_e) / (1.0 - p_e)


def run() -> list[dict]:
    rows = list(csv.DictReader(open(DATA / "judge_validation.csv", encoding="utf-8")))
    human = [r["human_label"] for r in rows]
    out = []
    for name, col in (("LLM judge (gemini-3.5-flash-lite)", "llm_judge_label"), ("Token rule", "token_rule_label")):
        pred = [r[col] for r in rows]
        agreed = sum(h == p for h, p in zip(human, pred))
        out.append({"grader": name, "n": len(rows), "agreed": agreed, "agreement": round(agreed / len(rows), 3),
                    "kappa": round(cohens_kappa(human, pred), 3),
                    "correct_agreement": round(sum((h == "correct") == (p == "correct") for h, p in zip(human, pred)) / len(rows), 3),
                    "stale_agreement": round(sum((h == "stale") == (p == "stale") for h, p in zip(human, pred)) / len(rows), 3)})
    return out
