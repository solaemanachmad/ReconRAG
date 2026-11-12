"""Human verification of the three-model consensus on full replacements (``data/replacement_verification.csv``).

One author checked a stratified random sample of 50 provisions on which all three annotators agree, seeing the
proposed status and successors (verification, not blind labelling). The table reports agreement with a Wilson 95%
interval, per status class, and the time spent per provision.
"""

from __future__ import annotations

import csv
import math
import statistics

from . import DATA


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def run() -> list[dict]:
    rows = list(csv.DictReader(open(DATA / "replacement_verification.csv", encoding="utf-8")))
    groups = [("all", rows)] + [(s, [r for r in rows if r["consensus_status"] == s])
                                for s in ("unchanged", "amended", "deleted", "non_substantive")]
    out = []
    for name, sub in groups:
        agreed = sum(r["human_agrees"] == "yes" for r in sub)
        lo, hi = wilson(agreed, len(sub))
        secs = [int(r["seconds"]) for r in sub]
        out.append({"status": name, "n": len(sub), "agreed": agreed, "agreement": round(agreed / len(sub), 3),
                    "wilson_low": round(lo, 3), "wilson_high": round(hi, 3),
                    "median_seconds": round(float(statistics.median(secs)), 1), "total_minutes": round(sum(secs) / 60, 1)})
    return out
