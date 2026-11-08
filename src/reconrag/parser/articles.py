"""Article and clause segmentation for Indonesian statutory texts."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_ARTICLE = re.compile(r"^\s*Pasal\s+(\d+[A-Z]?)\s*$")
_ROMAN = re.compile(r"^\s*Pasal\s+([IVX]+)\s*$")


@dataclass
class Article:
    number: str
    text: str
    clauses: dict[str, str] = field(default_factory=dict)


def split_articles(body: str) -> list[Article]:
    out: list[Article] = []
    cur: Article | None = None
    for line in body.split("\n"):
        m = _ARTICLE.match(line)
        if m:
            if cur:
                out.append(cur)
            cur = Article(number=m.group(1), text="")
        elif cur:
            cur.text += line + "\n"
    if cur:
        out.append(cur)
    for a in out:
        a.clauses = split_clauses(a.text)
    return out


def split_clauses(text: str) -> dict[str, str]:
    flat = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?:^|\s)\((\d+[a-z]?)\)\s", flat)
    clauses: dict[str, str] = {}
    for i in range(1, len(parts) - 1, 2):
        clauses.setdefault(parts[i], parts[i + 1].strip())
    return clauses


def numbering_gaps(articles: list[Article]) -> list[int]:
    """Return missing integer article numbers (indicative of unread headings in PDF extraction)."""
    nums = sorted({int(re.match(r"\d+", a.number).group()) for a in articles})
    return [n for n in range(1, (nums[-1] if nums else 0) + 1) if n not in nums]


def is_amendment_doc(text: str) -> bool:
    return bool(re.search(r"PERUBAHAN\s+(?:KEDUA\s+|KETIGA\s+)?ATAS\s+PERATURAN", text[:3000]))
