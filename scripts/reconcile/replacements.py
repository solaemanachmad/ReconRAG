"""Tier 1 Reconciliation Evaluation on 10 Full Replacement Pairs.

Evaluates 3-class status classification (unchanged / amended / deleted) on the
10 full-repeal replacement pairs (444 provisions) across all 4 silver references:
  - Three-model consensus (2/3 majority across 3 independent LLM judges)
  - Claude only           (Claude Sonnet 4.6)
  - Gemini only           (Gemini 3 Flash)
  - gpt-oss only          (GPT-OSS 120B)

Usage:
    python scripts/reconcile/replacements.py
"""

from __future__ import annotations

import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from reconrag import layer3  # noqa: E402
from reconrag.align import Params, global_ilp, per_provision, sim_matrix  # noqa: E402

DATA_DIR = ROOT / "data"
LABELS = ("unchanged", "amended", "deleted")
REFERENCES = [
    ("Three-model consensus", "silver_consensus_status", "silver_consensus_successors"),
    ("Claude only", "silver_claude_status", "silver_claude_successors"),
    ("Gemini only", "silver_gemini_status", "silver_gemini_successors"),
    ("gpt-oss only", "silver_gptoss_status", "silver_gptoss_successors"),
]


def _norm_silver_status(st: str) -> str:
    if st in ("unchanged", "non_substantive"):
        return "unchanged"
    if st in ("amended", "substantive"):
        return "amended"
    return "deleted"


def load_replacement_data() -> tuple[list[dict], list[dict]]:
    rows = [
        json.loads(line)
        for line in (DATA_DIR / "replacement_provisions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_pair[r["pair_id"]].append(r)

    load_doc = lambda reg_id: json.loads((DATA_DIR / "texts" / f"{reg_id}.json").read_text(encoding="utf-8"))
    pairs = []
    for pid, items in sorted(by_pair.items()):
        old_id = items[0]["old_regulation_id"]
        new_id = items[0]["new_regulation_id"]
        old_arts = [{"number": str(r["article_number"]), "text": r["old_text"]} for r in items]
        new_doc = load_doc(new_id)
        new_arts = new_doc["articles"]
        pairs.append({
            "pair_id": pid,
            "old_regulation_id": old_id,
            "new_regulation_id": new_id,
            "split": items[0]["split"],
            "old": old_arts,
            "new": new_arts,
            "items": items,
            "S": sim_matrix(old_arts, new_arts),
        })
    return pairs, rows


def score_macro_f1(P: dict[str, tuple[str, set]], G: dict[str, tuple[str, set]]) -> float:
    keys = list(G.keys())
    f1s = []
    for lab in LABELS:
        tp = sum(P[k][0] == lab and G[k][0] == lab for k in keys)
        fp = sum(P[k][0] == lab and G[k][0] != lab for k in keys)
        fn = sum(P[k][0] != lab and G[k][0] == lab for k in keys)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return sum(f1s) / len(LABELS)


def main() -> dict:
    print("=" * 88)
    print("TIER 1: FULL REPLACEMENT PAIRS ACROSS 4 SILVER REFERENCES")
    print("=" * 88)
    pairs, _ = load_replacement_data()
    l3_decisions = {
        r["provision_id"]: r["llm_decision"]
        for r in map(json.loads, open(DATA_DIR / "layer3_decisions.jsonl", encoding="utf-8"))
        if r["setting"] == "replacement"
    }

    taus = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    print(f"{'Silver Reference':<26} | {'Per-Prov + diff':>16} | {'Global ILP + diff':>18} | {'+ Layer 3 (Auto)':>18}")
    print("-" * 88)
    out = {}
    for ref_label, st_col, succ_col in REFERENCES:
        def run_split(solver, prm: Params, split: str):
            P, G = {}, {}
            for p in pairs:
                if p["split"] != split:
                    continue
                res = solver(p["old"], p["new"], p["S"], prm)
                for r in p["items"]:
                    if r[st_col] is None:
                        continue
                    art = str(r["article_number"])
                    key = r["provision_id"]
                    P[key] = (res.label[art], set(res.links[art]))
                    G[key] = (_norm_silver_status(r[st_col]), set(str(x) for x in r[succ_col]))
            return P, G

        best_pp = max(
            [Params(tau=t, label_mode="diff") for t in taus],
            key=lambda prm: score_macro_f1(*run_split(per_provision, prm, "dev")),
        )
        P_pp, G_test = run_split(per_provision, best_pp, "test")
        f1_pp = score_macro_f1(P_pp, G_test)

        best_ilp = max(
            [Params(tau=t, gamma_merge=g, gamma_split=g, label_mode="diff") for t, g in itertools.product(taus, [0.0, 0.05, 0.1])],
            key=lambda prm: score_macro_f1(*run_split(global_ilp, prm, "dev")),
        )
        P_ilp, _ = run_split(global_ilp, best_ilp, "test")
        f1_ilp = score_macro_f1(P_ilp, G_test)

        P_l3, _, _ = layer3.apply(P_ilp, l3_decisions)
        f1_l3 = score_macro_f1(P_l3, G_test)

        out[ref_label] = {"per_provision": f1_pp, "global_ilp": f1_ilp, "global_ilp_layer3": f1_l3}
        print(f"{ref_label:<26} | {f1_pp:16.3f} | {f1_ilp:18.3f} | {f1_l3:18.3f}")
    return out


if __name__ == "__main__":
    main()
