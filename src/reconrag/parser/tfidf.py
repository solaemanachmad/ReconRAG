"""Tokenization and TF-IDF cosine similarity for Indonesian regulatory provisions."""
from __future__ import annotations

import math
import re
from collections import Counter

_STOP = set(
    """dan atau yang di ke dari dalam pada untuk dengan sebagaimana dimaksud ayat huruf pasal ini itu
oleh atas adalah sebagai serta tidak dapat harus wajib paling sedikit banyak lain setiap tersebut berdasarkan
peraturan menteri""".split()
)


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP and len(t) > 2]


class Tfidf:
    def __init__(self, docs: list[str]):
        self.toks = [Counter(tokens(d)) for d in docs]
        df = Counter(t for c in self.toks for t in c)
        n = len(docs)
        self.idf = {t: math.log((1 + n) / (1 + f)) + 1 for t, f in df.items()}

    def vec(self, c: Counter) -> dict[str, float]:
        v = {t: (1 + math.log(k)) * self.idf.get(t, 1.0) for t, k in c.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}


def candidates(old: list[dict], new: list[dict], k: int = 3) -> list[dict]:
    """Return top-k candidate new articles per old article by TF-IDF cosine similarity."""
    tf = Tfidf([p["text"] for p in old] + [p["text"] for p in new])
    nv = [tf.vec(Counter(tokens(p["text"]))) for p in new]
    out = []
    for p in old:
        p_no = str(p.get("number", p.get("no")))
        ov = tf.vec(Counter(tokens(p["text"])))
        sims = [
            (sum(ov.get(t, 0) * x for t, x in v.items()), str(q.get("number", q.get("no"))))
            for v, q in zip(nv, new)
        ]
        sims.sort(reverse=True)
        out.append({"old": p_no, "cands": [{"number": no, "sim": round(s, 3)} for s, no in sims[:k]]})
    return out
