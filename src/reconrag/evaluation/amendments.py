"""Tier 1: provision-level reconciliation on the 47 amendment pairs (1,199 provisions), implicit setting.

Status classes unchanged / amended / deleted against official gold labels; parameters tuned on dev, scored on test.
"""

from __future__ import annotations

import csv
import itertools
import json
from collections import defaultdict

from .. import layer3
from ..align import Params, global_ilp, per_provision, sim_matrix
from ..parser.consolidate import consolidate
from . import DATA as DATA_DIR
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
        "n": len(keys),
        "macroF1": sum(f1.values()) / len(LABELS),
        "acc": sum(P[k][0] == G[k][0] for k in keys) / len(keys),
        "link_acc": sum(lk) / max(1, len(lk)),
        "del_F1": f1["deleted"],
        "stale_kept": sum(G[k][0] in ("amended", "deleted") and P[k][0] == "unchanged" for k in keys),
    }


def run() -> list[dict]:
    pairs = load_amendment_pairs()
    _ilp_cache: dict = {}

    def run_split(solver, prm: Params, split: str):
        P, G = {}, {}
        for p in pairs:
            if p["split"] != split:
                continue
            ck = (p["new_regulation_id"], solver.__name__, prm.tau, prm.gamma_merge, prm.gamma_split,
                  prm.label_mode if prm.label_mode == "diff" else prm.rho)
            if ck not in _ilp_cache:
                _ilp_cache[ck] = solver(p["old"], p["new"], p["S"], prm)
            res = _ilp_cache[ck]
            for k, lab in p["gold"].items():
                key = f"{p['new_regulation_id']}#{k}"
                P[key] = (res.label[k], set(res.links[k]))
                G[key] = (lab, p["succ"][k], p["domain"])
        return P, G

    taus = [0.1, 0.15, 0.2, 0.3, 0.4]
    rhos = [0.9, 0.95, 0.97, 0.99]
    methods = [
        ("Per-provision + character ratio", per_provision, [dict(tau=t, rho=r) for t, r in itertools.product(taus, rhos)]),
        ("Global ILP + character ratio", global_ilp,
         [dict(tau=t, rho=r, gamma_merge=g, gamma_split=g) for t, r, g in itertools.product(taus, rhos, [0.0, 0.1])]),
        ("Per-provision + conservative diff", per_provision, [dict(tau=t, label_mode="diff") for t in taus]),
        ("ReconRAG layers 1-2 (global ILP + diff)", global_ilp,
         [dict(tau=t, gamma_merge=g, gamma_split=g, label_mode="diff") for t, g in itertools.product(taus, [0.0, 0.1])]),
    ]
    rows, P, G = [], None, None
    rank_key = lambda s: (s["macroF1"], s["link_acc"])
    for name, solver, grid in methods:
        best = max(grid, key=lambda g: rank_key(score(*run_split(solver, Params(**g), "dev"))))
        P, G = run_split(solver, Params(**best), "test")
        rows.append(_row(name, score(P, G)))

    ans = {r["provision_id"]: r["llm_decision"]
           for r in map(json.loads, open(DATA_DIR / "layer3_decisions.jsonl", encoding="utf-8"))
           if r["setting"] == "amendment_implicit"}
    P_auto, _, _ = layer3.apply(P, ans)
    rows.append(_row("ReconRAG + layer 3, automatic", score(P_auto, G)))
    P_safe, calls, esc = layer3.apply_escalate(P, ans, G)
    rows.append(_row("ReconRAG + layer 3, safe policy", score(P_safe, G), escalated=esc, calls=calls))
    return rows


def _row(name: str, s: dict, escalated: int | None = None, calls: int | None = None) -> dict:
    return {"method": name, "n": s["n"], "macro_f1": round(s["macroF1"], 3), "accuracy": round(s["acc"], 3),
            "link_accuracy": round(s["link_acc"], 3), "deleted_f1": round(s["del_F1"], 3),
            "stale_kept": s["stale_kept"], "escalated": "" if escalated is None else escalated,
            "llm_calls": "" if calls is None else calls}
