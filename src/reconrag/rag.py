"""ReconRAG version-reconciled index, retrievers, status tagging, and successor expansion."""
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

from .parser.consolidate import deleted_articles, restated_articles
from .parser.tfidf import tokens

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"


def _load_doc(reg_id: str) -> dict:
    return json.loads((DATA_DIR / "texts" / f"{reg_id}.json").read_text(encoding="utf-8"))


_PAGE = re.compile(r"(?m)^\s*\d{4}\s*,\s*No\.\s*\d+(?:\s+-?\s*\d+\s*-?)?\s*$\n?")


def tidy(t: str) -> str:
    """Strip residual official gazette page headers ('2015, No.1372 5') from article texts."""
    return _PAGE.sub("", t).strip()


def doc_year(doc_id: str) -> int:
    return int(doc_id.rsplit("-", 1)[1])


def build_archive() -> tuple[list[dict], list[dict]]:
    """Build the as-published regulation archive (1,457 provision chunks across 47 amendment families)."""
    fams, chunks = [], []
    for r in csv.DictReader(open(DATA_DIR / "regulations_metadata.csv", encoding="utf-8")):
        if r["subset"] != "amendment":
            continue
        b, m = r["old_regulation_id"], r["new_regulation_id"]
        tb = re.sub(r"\s*-\s*JDIH BPK$", "", r["old_title"] or b)
        tm = (r["new_title"] or m).split(" - Perubahan Atas ")[0].strip()
        base_doc = _load_doc(b)
        amd_doc = _load_doc(m)
        amd_text = amd_doc.get("full_text", "")
        fams.append({
            "family_id": m,
            "base_id": b,
            "amendment_id": m,
            "split": r["split"],
            "domain": r["domain"],
            "base_title": tb,
            "amendment_title": tm,
        })
        for p in base_doc["articles"]:
            chunks.append({
                "id": f"{b}#{p['number']}",
                "family_id": m,
                "doc_id": b,
                "kind": "base",
                "article": str(p["number"]),
                "text": tidy(p["text"]),
                "title": tb,
                "year": doc_year(b),
            })
        for no, txt in restated_articles(amd_text).items():
            chunks.append({
                "id": f"{m}#{no}",
                "family_id": m,
                "doc_id": m,
                "kind": "amendment",
                "article": str(no),
                "text": tidy(txt),
                "title": tm,
                "year": doc_year(m),
            })
        for no in deleted_articles(amd_text):
            chunks.append({
                "id": f"{m}#del{no}",
                "family_id": m,
                "doc_id": m,
                "kind": "amendment",
                "article": f"del{no}",
                "text": f"Pasal {no} {tb} dihapus.",
                "title": tm,
                "year": doc_year(m),
                "deletes": str(no),
            })
    return chunks, fams


class Retriever:
    """TF-IDF cosine retriever over document titles and provision texts."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        docs = [Counter(tokens(c["title"] + " " + c["text"])) for c in chunks]
        df = Counter(t for d in docs for t in d)
        n = len(docs)
        self.idf = {t: math.log((n + 1) / (f + 1)) + 1 for t, f in df.items()}
        self.vecs = [self._vec(d) for d in docs]

    def _vec(self, c: Counter) -> dict[str, float]:
        v = {t: (1 + math.log(x)) * self.idf.get(t, 0.0) for t, x in c.items()}
        nrm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / nrm for t, x in v.items()}

    def search(self, q: str, k: int = 5, allow=None) -> list[tuple[float, int]]:
        qv = self._vec(Counter(tokens(q)))
        sc = []
        for i, v in enumerate(self.vecs):
            if allow is not None and not allow(self.chunks[i]):
                continue
            s = sum(x * v.get(t, 0.0) for t, x in qv.items())
            if s > 0:
                sc.append((s, i))
        return sorted(sc, reverse=True)[:k]


def reconcile_family(chunks: list[dict], fam: str, solver, params) -> dict[str, dict]:
    """Run ingest-time reconciliation for one regulation family without using article numbers."""
    from .align import sim_matrix

    old = [c for c in chunks if c["family_id"] == fam and c["kind"] == "base"]
    new = [c for c in chunks if c["family_id"] == fam and c["kind"] == "amendment" and "deletes" not in c]
    dels = {c["deletes"] for c in chunks if c["family_id"] == fam and "deletes" in c}
    out = {}
    if new:
        oo = [{"number": c["id"], "text": c["text"]} for c in old]
        nn = [{"number": c["id"], "text": c["text"]} for c in new]
        res = solver(oo, nn, sim_matrix(oo, nn), params)
        for c in old:
            lab, links = res.label[c["id"]], res.links[c["id"]]
            out[c["id"]] = {"status": "amended" if lab == "amended" else "unchanged", "successors": links}
    for c in old:
        out.setdefault(c["id"], {"status": "unchanged", "successors": []})
        if c["article"] in dels:
            out[c["id"]] = {"status": "deleted", "successors": []}
    return out


def gold_family(chunks: list[dict], fam_rec: dict, gold_by_prov: dict[str, str]) -> dict[str, dict]:
    out = {}
    for c in chunks:
        if c["family_id"] == fam_rec["family_id"] and c["kind"] == "base":
            st = gold_by_prov.get(c["id"], "unchanged")
            succ = [f"{fam_rec['amendment_id']}#{c['article']}"] if st == "amended" else []
            out[c["id"]] = {"status": st, "successors": succ}
    return out


def render(c: dict, mode: str, dec: dict | None = None, fam_titles: dict | None = None) -> tuple[str, str]:
    """Render a retrieved provision with its version status tag ('in_force' | 'superseded' | 'deleted')."""
    if mode == "Naive_RAG":
        return c["text"], "in_force"
    head = (
        f"[{c['title']}] Pasal {c['article']}"
        if c["kind"] == "base"
        else f"[{c['title']}] Pasal {c['article']} sebagaimana diubah"
        if "deletes" not in c
        else f"[{c['title']}]"
    )
    if mode in ("Recency_Prompt", "Latest_Doc"):
        return f"{head}:\n{c['text']}", "in_force"
    d = (dec or {}).get(c["id"])
    if d and d["status"] == "amended":
        return f"{head} [VERSI LAMA — TIDAK BERLAKU, telah diubah oleh {fam_titles[c['family_id']]}]:\n{c['text']}", "superseded"
    if d and d["status"] == "deleted":
        return f"{head} [DIHAPUS oleh {fam_titles[c['family_id']]} — TIDAK BERLAKU]:\n{c['text']}", "deleted"
    return f"{head} [BERLAKU]:\n{c['text']}", "in_force"


CONDITIONS = [
    "Naive_RAG",
    "Recency_Prompt",
    "Latest_Doc",
    "ReconRAG_StatusOnly",
    "ReconRAG_FilterOnly",
    "ReconRAG",
    "ReconRAG_Route",
    "Oracle_StatusOnly",
    "Oracle",
]

CONDITION_ALIASES: dict[str, str] = {
    "B0": "Naive_RAG",
    "Naive RAG": "Naive_RAG",
    "B1": "Recency_Prompt",
    "Recency prompt": "Recency_Prompt",
    "B2": "Latest_Doc",
    "Latest doc": "Latest_Doc",
    "R": "ReconRAG_StatusOnly",
    "ReconRAG (status only)": "ReconRAG_StatusOnly",
    "R-filter": "ReconRAG_FilterOnly",
    "R+succ": "ReconRAG",
    "ReconRAG (Ours)": "ReconRAG",
    "R+route": "ReconRAG_Route",
    "ReconRAG+Route": "ReconRAG_Route",
    "O": "Oracle_StatusOnly",
    "O+succ": "Oracle",
    "Oracle ceiling": "Oracle",
}

CONDITION_LABELS: dict[str, str] = {
    "Naive_RAG": "Naive RAG",
    "Recency_Prompt": "Recency prompt",
    "Latest_Doc": "Latest doc",
    "Query_Time_Judge": "Query-time judge",
    "ReconRAG_StatusOnly": "ReconRAG (status only)",
    "ReconRAG_FilterOnly": "ReconRAG (filter only)",
    "ReconRAG": "ReconRAG (Ours)",
    "ReconRAG_Route": "ReconRAG + Route (ablation)",
    "Oracle_StatusOnly": "Oracle (status only)",
    "Oracle": "Oracle ceiling",
}

HIST_CUE = re.compile(
    r"sebelum (?:diubah|perubahan|dicabut)|ketentuan lama|aturan lama|versi lama|semula|\b(?:19|20)\d{2}\b",
    re.I,
)


class ReconRAGBench:
    """ReconRAG benchmark archive, ingest-time reconciliation index, and query-time context builder."""

    def __init__(self, k: int = 5, retriever=None):
        from .align import Params, global_ilp

        self.params = Params(tau=0.3, cap_old=1, cap_new=1, gamma_merge=0.1, gamma_split=0.1, label_mode="diff")
        self.k = k
        self.chunks, self.fams = build_archive()
        self.titles = {f["family_id"]: f["amendment_title"] for f in self.fams}
        gold_by_prov = {
            r["provision_id"]: r["gold_status"]
            for r in map(json.loads, open(DATA_DIR / "amendment_provisions.jsonl", encoding="utf-8"))
        }
        self.dec_R, self.dec_O = {}, {}
        for f in self.fams:
            self.dec_R.update(reconcile_family(self.chunks, f["family_id"], global_ilp, self.params))
            self.dec_O.update(gold_family(self.chunks, f, gold_by_prov))
        self.ret = retriever if retriever is not None else Retriever(self.chunks)
        self.by_id = {c["id"]: i for i, c in enumerate(self.chunks)}

    def _expand(self, hits, dec):
        """Successor expansion: prepend stored in-force successors ahead of retrieved superseded provisions."""
        out = []
        for s_, i in hits:
            d = dec.get(self.chunks[i]["id"])
            if d and d["status"] == "amended":
                out += [
                    (s_, self.by_id[b])
                    for b in d["successors"]
                    if b in self.by_id and all(self.by_id[b] != j for _, j in out)
                ]
            if all(i != j for _, j in out):
                out.append((s_, i))
        return out[: self.k]

    def context(self, question: str, cond: str = "ReconRAG") -> list[dict]:
        cond = CONDITION_ALIASES.get(cond, cond)
        if cond == "ReconRAG_Route":
            cond = "ReconRAG" if HIST_CUE.search(question) else "ReconRAG_FilterOnly"
        dec = self.dec_R if cond.startswith("ReconRAG") else self.dec_O if cond.startswith("Oracle") else None
        allow = None
        if cond == "Latest_Doc":
            allow = lambda c: c["kind"] == "amendment"
        elif cond == "ReconRAG_FilterOnly":
            allow = lambda c: self.dec_R.get(c["id"], {"status": "unchanged"})["status"] == "unchanged"
        hits = self.ret.search(question, self.k, allow)
        if cond in ("ReconRAG", "Oracle"):
            hits = self._expand(hits, dec)
        mode = "ReconRAG" if dec is not None else cond
        out = []
        for _, i in hits:
            txt, st = render(self.chunks[i], mode, dec, self.titles)
            out.append({"id": self.chunks[i]["id"], "text": txt, "raw": self.chunks[i]["text"], "status": st})
        return out


Bench = ReconRAGBench
