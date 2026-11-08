"""Parser for official Indonesian legislative amendment instructions (Pasal I, Law 12/2011 Annex II)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_INSTR = re.compile(
    r"(?:^|\s)(\d+)\.\s+((?:Ketentuan\s+)?(?:Pasal|ayat|huruf|Di\s*antara|Penjelasan|Lampiran|BAB)"
    r"[^:]*?(?:diubah|disisipkan|dihapus|ditambah)[^:]*?)(?=\s*(?:sehingga|sebagai berikut|berbunyi|:|\.\s|$))",
    re.IGNORECASE,
)
OPS = ("diubah", "disisipkan", "dihapus", "ditambah")


@dataclass
class Instruction:
    index: int
    text: str
    articles: list[str] = field(default_factory=list)
    clauses: list[str] = field(default_factory=list)
    ops: list[str] = field(default_factory=list)


def article_i_body(text: str) -> str:
    """Extract the body of Pasal I (the numbered list of amendment instructions)."""
    m = re.search(r"\n\s*Pasal\s+I\s*\n(.*?)(?:\n\s*Pasal\s+II\s*\n|$)", text, re.S)
    return m.group(1) if m else text


def parse_instructions(text: str) -> list[Instruction]:
    flat = re.sub(r"\s+", " ", article_i_body(text))
    out = []
    for m in _INSTR.finditer(flat):
        t = m.group(2).strip()
        out.append(
            Instruction(
                index=int(m.group(1)),
                text=t,
                articles=re.findall(r"Pasal\s+(\d+[A-Z]?)", t),
                clauses=re.findall(r"ayat\s+\((\d+[a-z]?)\)", t),
                ops=[o for o in OPS if re.search(o, t, re.I)],
            )
        )
    return out


def target_article(ins: Instruction) -> str | None:
    """Return the base article modified by `ins`, or None if `ins` inserts a new article between two articles."""
    if re.match(r"Di\s*antara\s+(?:Pasal|Bab)\b", ins.text, re.I):
        return None
    return ins.articles[0] if ins.articles else None


def inserted_articles(ins: Instruction) -> list[str]:
    """Return newly inserted article numbers (e.g. ['83A', '83B', '83C'])."""
    if not re.match(r"Di\s*antara\s+Pasal", ins.text, re.I):
        return []
    return [p for p in ins.articles if re.search(r"[A-Z]$", p)]


def gold_after_amendment(old_article_numbers: list[str], instrs: list[Instruction]) -> dict[str, str]:
    """Derive official gold provision labels ('unchanged', 'amended', 'deleted') from amendment instructions."""
    gold = {no: "unchanged" for no in old_article_numbers}
    for ins in instrs:
        t = target_article(ins)
        if t is None:
            continue
        whole_deleted = bool(re.match(r"(?:Ketentuan\s+)?Pasal\s+\d+[A-Z]?\s+dihapus", ins.text, re.I))
        gold[t] = "deleted" if whole_deleted else "amended"
    return gold
