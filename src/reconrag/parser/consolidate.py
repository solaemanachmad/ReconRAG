"""Consolidation of base regulations with official amendment instructions."""
from __future__ import annotations

import re

from .amendments import article_i_body
from .articles import split_clauses

_INSTR_START = re.compile(
    r"(?m)^\s*\d+\.\s+(?:Ketentuan|Mengubah|Menambah|Menyisipkan|Menghapus|Di\s*antara|Diantara|"
    r"Pasal\s+\d+[A-Z]?\s+dihapus|BAB|Lampiran|Penjelasan)"
)
_HEAD = re.compile(r"(?m)^\s*Pasal\s+(\d+[A-Za-z]?)\s*$")


def restated_articles(amend_text: str) -> dict[str, str]:
    """Return article numbers and their restated texts from Pasal I of an amending regulation."""
    body = article_i_body(amend_text)
    starts = [m.start() for m in _INSTR_START.finditer(body)]
    if not starts:
        starts = [0]
    starts += [len(body)]
    out: dict[str, str] = {}
    for a, b in zip(starts, starts[1:]):
        seg = body[a:b]
        heads = list(_HEAD.finditer(seg))
        for i, h in enumerate(heads):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(seg)
            out[h.group(1).upper()] = seg[h.end() : end].strip() + "\n"
    return out


def deleted_articles(amend_text: str) -> list[str]:
    body = re.sub(r"\s+", " ", article_i_body(amend_text))
    return [
        n.upper()
        for n in re.findall(r"(?:^|\d+\.\s+|Pasal I\s+)(?:Ketentuan\s+)?Pasal\s+(\d+[A-Za-z]?)\s+dihapus", body, re.I)
    ]


def consolidate(base_articles: list[dict], amend_text: str) -> list[dict]:
    """Consolidate `base_articles` ([{'number', 'text'}]) with `amend_text`."""
    new = restated_articles(amend_text)
    gone = set(deleted_articles(amend_text))
    out, seen = [], set()
    for p in base_articles:
        no = str(p.get("number", p.get("no")))
        if no in gone:
            continue
        txt = new.get(no, p["text"])
        out.append({"number": no, "text": txt, "clauses": split_clauses(txt), "amended": no in new})
        seen.add(no)

    def sort_key(no: str):
        m = re.match(r"(\d+)([A-Z]?)", no)
        return (int(m.group(1)), m.group(2))

    for no, txt in new.items():
        if no not in seen:
            out.append({"number": no, "text": txt, "clauses": split_clauses(txt), "amended": True, "inserted": True})
    return sorted(out, key=lambda p: sort_key(p["number"]))
