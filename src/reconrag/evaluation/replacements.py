"""Tier 1: reconciliation on the 10 full-replacement pairs (444 old provisions) against four silver references.

Definitions (as in the paper):

* Four status classes: unchanged, amended, deleted, non_rule (definitions, commencement and repeal clauses).
* Old provisions are grouped by their NEW regulation (one ILP per new regulation), because several old
  regulations can be replaced by one new regulation.
* Split by family of new regulations (dev: 3 new regulations, test: 4), never by row.
* Silver references:
    - consensus: all three annotators agree on the status and, for unchanged/amended, share at least one
      successor (successor set = intersection); test n = 275
    - claude / gemini / gptoss: that annotator's status and successors; test n = 366
* Per-provision and global matching are tuned on dev separately for each reference (key: macro-F1, link acc.).
* Layer 3 uses fixed parameters (tau=0.2, gamma=0.2, diff labelling; tuned on dev with the consensus) and the
  stored Gemini decisions. "safe" = LLM may only confirm 'amended'; every other decision is escalated and the
  curator is simulated by the reference label. "auto" = LLM flips are applied directly.
* Macro-F1 averages over classes that occur in the reference.
"""

from __future__ import annotations

import itertools
import json
import re
from collections import defaultdict

from .. import layer3
from ..align import Params, global_ilp, per_provision, sim_matrix
from . import DATA as DATA_DIR

LABELS = ("unchanged", "amended", "deleted", "non_rule")
STATUS = {"unchanged": "unchanged", "amended": "amended", "deleted": "deleted", "non_substantive": "non_rule"}
ANNOTATORS = ("gemini", "gptoss", "claude")
REFERENCES = [("Three-model consensus", "consensus"), ("Claude", "claude"),
              ("Gemini (also layer 3)", "gemini"), ("gpt-oss", "gptoss")]
TAUS = [0.15, 0.2, 0.25, 0.3, 0.35, 0.45]
GAMMAS = [0.0, 0.05, 0.1, 0.2]
PRM_L3 = Params(tau=0.2, gamma_merge=0.2, gamma_split=0.2, label_mode="diff")
NON_RULE = re.compile(r"(mulai berlaku pada tanggal diundangkan|agar setiap orang mengetahuinya|"
                      r"dicabut dan dinyatakan tidak berlaku|tetap berlaku sepanjang tidak bertentangan)", re.I)


def _vote(row: dict, who: str):
    st = row.get(f"silver_{who}_status")
    if st not in STATUS:
        return None
    return STATUS[st], {str(x) for x in (row.get(f"silver_{who}_successors") or [])}


def silver_of(row: dict, ref: str):
    if ref != "consensus":
        return _vote(row, ref)
    votes = [_vote(row, w) for w in ANNOTATORS]
    if any(v is None for v in votes) or len({v[0] for v in votes}) != 1:
        return None
    lab, links = votes[0][0], set.intersection(*(v[1] for v in votes))
    if lab in ("unchanged", "amended") and not links:
        return None
    return lab, links


def load_groups() -> list[dict]:
    rows = [json.loads(l) for l in (DATA_DIR / "replacement_provisions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_new: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_new[r["new_regulation_id"]].append(r)
    groups = []
    for new_id, items in sorted(by_new.items()):
        splits = {r["split"] for r in items}
        assert len(splits) == 1, f"mixed split for {new_id}"
        old = [{"number": r["provision_id"], "text": r["old_text"]} for r in items]
        new = json.loads((DATA_DIR / "texts" / f"{new_id}.json").read_text(encoding="utf-8"))["articles"]
        groups.append({"new_id": new_id, "split": splits.pop(), "items": items, "old": old, "new": new,
                       "S": sim_matrix(old, new)})
    return groups


def run_pred(groups, solver, prm: Params, split: str, ref: str):
    pred, gold = {}, {}
    for g in groups:
        if g["split"] != split:
            continue
        res = solver(g["old"], g["new"], g["S"], prm)
        for r in g["items"]:
            k = r["provision_id"]
            lab, links = res.label[k], set(res.links[k])
            if NON_RULE.search(r["old_text"]) and len(r["old_text"]) < 1500:
                lab, links = "non_rule", set()
            sv = silver_of(r, ref)
            if sv:
                pred[k], gold[k] = (lab, links), sv
    return pred, gold


def score(pred: dict, gold: dict) -> dict:
    f1s = []
    for lab in LABELS:
        tp = sum(pred[k][0] == lab and gold[k][0] == lab for k in gold)
        fp = sum(pred[k][0] == lab and gold[k][0] != lab for k in gold)
        fn = sum(pred[k][0] != lab and gold[k][0] == lab for k in gold)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        if tp + fn:
            f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    lk = [bool(pred[k][1] & gold[k][1]) for k in gold if gold[k][0] in ("unchanged", "amended") and gold[k][1]]
    return {"n": len(gold), "macro_f1": sum(f1s) / len(f1s),
            "link_acc": sum(lk) / max(1, len(lk)),
            "stale_kept": sum(gold[k][0] in ("amended", "deleted") and pred[k][0] == "unchanged" for k in gold)}


def tune(groups, solver, grid, ref):
    return max(grid, key=lambda prm: (lambda s: (round(s["macro_f1"], 3), round(s["link_acc"], 3)))(
        score(*run_pred(groups, solver, prm, "dev", ref))))


def run() -> list[dict]:
    groups = load_groups()
    l3 = {r["provision_id"]: r["llm_decision"]
          for r in map(json.loads, open(DATA_DIR / "layer3_decisions.jsonl", encoding="utf-8"))
          if r["setting"] == "replacement"}
    grid_pp = [Params(tau=t, label_mode="diff") for t in TAUS]
    grid_ilp = [Params(tau=t, gamma_merge=g, gamma_split=g, label_mode="diff") for t, g in itertools.product(TAUS, GAMMAS)]
    rows = []
    for name, ref in REFERENCES:
        s_pp = score(*run_pred(groups, per_provision, tune(groups, per_provision, grid_pp, ref), "test", ref))
        s_ilp = score(*run_pred(groups, global_ilp, tune(groups, global_ilp, grid_ilp, ref), "test", ref))
        P, G = run_pred(groups, global_ilp, PRM_L3, "test", ref)
        P_safe, calls, esc = layer3.apply_escalate(P, l3, G)
        P_auto, _, _ = layer3.apply(P, l3)
        s_safe, s_auto = score(P_safe, G), score(P_auto, G)
        rows.append({"reference": name, "n": s_pp["n"],
                     "per_provision": round(s_pp["macro_f1"], 3), "global_ilp": round(s_ilp["macro_f1"], 3),
                     "layer3_safe": round(s_safe["macro_f1"], 3), "layer3_auto": round(s_auto["macro_f1"], 3),
                     "stale_safe": s_safe["stale_kept"], "stale_auto": s_auto["stale_kept"],
                     "escalated": esc, "escalated_share": round(esc / s_pp["n"], 3), "llm_calls": calls})
    return rows
