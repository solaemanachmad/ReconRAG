"""LLM Judge 5-class evaluation rubric and response parser."""
from __future__ import annotations

import json

_LABEL_NORM = {
    "benar": "correct",
    "correct": "correct",
    "usang": "stale",
    "stale": "stale",
    "campur": "mixed",
    "mixed": "mixed",
    "abstain": "abstain",
    "lain": "other",
    "other": "other",
}


def parse(raw: str) -> str:
    dec = json.JSONDecoder()
    for i, ch in enumerate(raw or ""):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(raw[i:])
                if isinstance(obj, dict):
                    val = obj.get("nilai") or obj.get("label")
                    if val in _LABEL_NORM:
                        return _LABEL_NORM[val]
            except json.JSONDecodeError:
                continue
    return "other"
