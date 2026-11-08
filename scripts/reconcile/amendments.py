"""Tier 1 Reconciliation Evaluation on 47 Amendment Pairs.

Evaluates 3-class status classification (unchanged / amended / deleted) and
successor link accuracy on the 47 amendment pairs (1,199 provisions) using the
dev/test split defined in ``data/regulations_metadata.csv``.

Usage:
    python scripts/reconcile/amendments.py
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from reconrag import layer3  # noqa: E402
from reconrag.align import Params, global_ilp, per_provision, sim_matrix  # noqa: E402
from reconrag.parser.consolidate import consolidate  # noqa: E402

DATA_DIR = ROOT / "data"
LABELS = ("unchanged", "amended", "deleted")


def load_amendment_pairs() -> list[dict]:
    load_doc = lambda reg_id: json.loads((DATA_DIR / "texts" / f"{reg_id}.json").read_text(encoding="utf-8"))

    gold_by_pair: dict[str, dict[str, str]] = defaultdict(dict)
    succ_by_pair: dict[str, dict[str, set[str]]] = defaultdict(dict)
    for line in (DATA_DIR / "amendment_provisions.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        pid = r["pair_id"]
        art = str(r["article_number"])
        gold_by_pair[pid][art] = r["gold_status"]
        succ_by_pair[pid][art] = set(str(s) for s in r.get("successor_articles", []))

    pairs = []
    with (DATA_DIR / "regulations_metadata.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["subset"] != "amendment":
                continue
            pid = r["pair_id"]
            old_doc = load_doc(r["old_regulation_id"])
            new_doc = load_doc(r["new_regulation_id"])
            old_arts = [a for a in old_doc["articles"] if str(a["number"]) in gold_by_pair[pid]]
            new_arts = consolidate(old_doc["articles"], new_doc.get("full_text", ""))
            pairs.append({
                **r,
                "old": old_arts,
                "new": new_arts,
                "gold": gold_by_pair[pid],
                "succ": succ_by_pair[pid],
                "S": sim_matrix(old_arts, new_arts),
            })
    return pairs


def score(P: dict, G: dict) -> dict:
    keys = list(G.keys())
    f1 = {}
    for lab in LABELS:
        tp = sum(P[k][0] == lab and G[k][0] == lab for k in keys)
        fp = sum(P[k][0] == lab and G[k][0] != lab for k in keys)
        fn = sum(P[k][0] != lab and G[k][0] == lab for k in keys)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1[lab] = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    lk = [bool(P[k][1] & G[k][1]) for k in keys if G[k][0] != "deleted" and G[k][1]]
    return {
        "macroF1": sum(f1.values()) / len(LABELS),
        "acc": sum(P[k][0] == G[k][0] for k in keys) / len(keys),
        "link_acc": sum(lk) / max(1, len(lk)),
        "del_F1": f1["deleted"],
        "stale_kept": sum(G[k][0] in ("amended", "deleted") and P[k][0] == "unchanged" for k in keys),
    }


def main() -> dict:
    print("=" * 88)
    print("1. TIER 1: PROVISION-LEVEL RECONCILIATION ON AMENDMENTS")
    print("=" * 88)
    pairs = load_amendment_pairs()
    _ilp_cache: dict = {}

    def run_split(solver, prm: Params, split: str):
        P, G = {}, {}
        for p in pairs:
            if p["split"] != split:
                continue
            ck = (
                p["new_regulation_id"],
                solver.__name__,
                prm.tau,
                prm.gamma_merge,
                prm.gamma_split,
                prm.label_mode if prm.label_mode == "diff" else prm.rho,
            )
            if ck not in _ilp_cache:
                _ilp_cache[ck] = solver(p["old"], p["new"], p["S"], prm)
            res = _ilp_cache[ck]
            for k, lab in p["gold"].items():
                key = f"{p['new_regulation_id']}#{k}"
                P[key] = (res.label[k], set(res.links[k]))
                G[key] = (lab, p["succ"][k], p["domain"])
        return P, G

    taus = [0.1, 0.15, 0.2, 0.3, 0.4]
    methods = [
        (
            "Per-provision + char. ratio",
            per_provision,
            [dict(tau=t, rho=r) for t, r in itertools.product(taus, [0.9, 0.95, 0.97, 0.99])],
        ),
        (
            "Global (ILP) + char. ratio",
            global_ilp,
            [
                dict(tau=t, rho=r, gamma_merge=g, gamma_split=g)
                for t, r, g in itertools.product(taus, [0.9, 0.95, 0.97, 0.99], [0.0, 0.1])
            ],
        ),
        ("Per-provision + diff", per_provision, [dict(tau=t, label_mode="diff") for t in taus]),
        (
            "ReconRAG Layer 1+2 (Global + diff)",
            global_ilp,
            [dict(tau=t, gamma_merge=g, gamma_split=g, label_mode="diff") for t, g in itertools.product(taus, [0.0, 0.1])],
        ),
    ]

    print(f"{'Method':<36} | {'Macro-F1':>8} | {'Accuracy':>8} | {'Link Acc':>8} | {'Del F1':>6} | {'Stale Kept':>10}")
    print("-" * 88)
    last_P, last_G = None, None
    results = {}
    rank_key = lambda s: (s["macroF1"], s["link_acc"])
    for name, solver, grid in methods:
        best = max(grid, key=lambda g: rank_key(score(*run_split(solver, Params(**g), "dev"))))
        P, G = run_split(solver, Params(**best), "test")
        s = score(P, G)
        results[name] = s
        last_P, last_G = P, G
        print(
            f"{name:<36} | {s['macroF1']:8.3f} | {s['acc']:8.3f} | {s['link_acc']:8.3f} | "
            f"{s['del_F1']:6.2f} | {s['stale_kept']:10d}"
        )

    ans_f = DATA_DIR / "layer3_decisions.jsonl"
    if ans_f.exists() and last_P is not None and last_G is not None:
        ans = {
            r["provision_id"]: r["llm_decision"]
            for r in map(json.loads, open(ans_f, encoding="utf-8"))
            if r["setting"] == "amendment_implicit"
        }
        P_auto, _, _ = layer3.apply(last_P, ans)
        s_auto = score(P_auto, last_G)
        results["Layer 3 (automatic)"] = s_auto
        print(
            f"{'  + Layer 3 (automatic)':<36} | {s_auto['macroF1']:8.3f} | {s_auto['acc']:8.3f} | "
            f"{s_auto['link_acc']:8.3f} | {s_auto['del_F1']:6.2f} | {s_auto['stale_kept']:10d}"
        )
        P_safe, _, esc = layer3.apply_escalate(last_P, ans, last_G)
        s_safe = score(P_safe, last_G)
        results["Layer 3 (safe policy)"] = s_safe
        print(
            f"{'  + Layer 3 (safe policy, esc=' + str(esc) + ')':<36} | {s_safe['macroF1']:8.3f} | "
            f"{s_safe['acc']:8.3f} | {s_safe['link_acc']:8.3f} | {s_safe['del_F1']:6.2f} | {s_safe['stale_kept']:10d}"
        )
    return results


if __name__ == "__main__":
    main()
