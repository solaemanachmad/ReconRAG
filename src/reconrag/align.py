"""Layer 1 (Globally Constrained ILP Matching) and Layer 2 (Conservative Diff Change Detection).

Given old provisions O = {o_1, ..., o_m} and new provisions N = {n_1, ..., n_k}, outputs for each o_i:
  - label in {"unchanged", "amended", "deleted"}
  - successor links in N
"""
from __future__ import annotations

import difflib
import functools
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .parser.tfidf import Tfidf, tokens


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t.lower())).strip()


@functools.lru_cache(maxsize=None)
def char_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _norm(a), _norm(b), autojunk=False).ratio()


def sim_matrix(old: list[dict], new: list[dict]) -> np.ndarray:
    tf = Tfidf([p["text"] for p in old] + [p["text"] for p in new])
    ov = [tf.vec(Counter(tokens(p["text"]))) for p in old]
    nv = [tf.vec(Counter(tokens(p["text"]))) for p in new]
    S = np.zeros((len(old), len(new)))
    for i, a in enumerate(ov):
        for j, b in enumerate(nv):
            S[i, j] = sum(a.get(t, 0.0) * x for t, x in b.items())
    return S


_WM = re.compile(r"peraturan\s+bpk\s+go\s+id(\s+\d{4}\s+no\s+\d+)?")
_EDITORIAL = re.compile(r"^(\d+[a-z]?|[a-z]|[ivxlc]+|dan|atau|pada|dalam|di|ke|yang)$")


@functools.lru_cache(maxsize=None)
def substantive_diff(a: str, b: str) -> bool:
    """Return True if the difference between `a` and `b` contains a substantive change.

    Differences consisting solely of shifted numbers, list enumerators, Roman numerals, short connectives,
    or known watermarks are treated as editorial (False).
    """
    ta = _WM.sub(" ", _norm(a)).split()
    tb = _WM.sub(" ", _norm(b)).split()
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes():
        if op != "equal" and not all(_EDITORIAL.match(t) for t in ta[i1:i2] + tb[j1:j2]):
            return True
    return False


@dataclass
class Params:
    tau: float = 0.30
    rho: float = 0.97
    cap_old: int = 2
    cap_new: int = 3
    second_link_margin: float = 0.15
    topk: int = 8
    label_mode: str = "ratio"  # "ratio" | "diff"
    gamma_merge: float = 0.0
    gamma_split: float = 0.0


@dataclass
class Result:
    label: dict[str, str] = field(default_factory=dict)
    links: dict[str, list[str]] = field(default_factory=dict)
    new_unmatched: list[str] = field(default_factory=list)


def _id_of(item: dict) -> str:
    return str(item.get("number", item.get("no", item.get("id"))))


def _labels(old: list[dict], new: list[dict], links_idx: dict[int, list[int]], p: Params) -> Result:
    r = Result()
    used = set()
    for i, o in enumerate(old):
        oid = _id_of(o)
        js = links_idx.get(i, [])
        used |= set(js)
        r.links[oid] = [_id_of(new[j]) for j in js]
        if not js:
            r.label[oid] = "deleted"
        else:
            if p.label_mode == "diff":
                same = any(not substantive_diff(o["text"], new[j]["text"]) for j in js)
            else:
                same = max(char_ratio(o["text"], new[j]["text"]) for j in js) >= p.rho
            r.label[oid] = "unchanged" if same else "amended"
    r.new_unmatched = [_id_of(new[j]) for j in range(len(new)) if j not in used]
    return r


def per_provision(old: list[dict], new: list[dict], S: np.ndarray, p: Params) -> Result:
    links = {}
    for i in range(len(old)):
        j = int(S[i].argmax())
        if S[i, j] >= p.tau:
            links[i] = [j]
    return _labels(old, new, links, p)


_ILP_LINKS_CACHE: dict[tuple, dict[int, list[int]]] = {}


def global_ilp(old: list[dict], new: list[dict], S: np.ndarray, p: Params) -> Result:
    cache_key = (
        id(S),
        S.shape,
        p.tau,
        p.cap_old,
        p.cap_new,
        p.second_link_margin,
        p.topk,
        p.gamma_merge,
        p.gamma_split,
    )
    if cache_key in _ILP_LINKS_CACHE:
        return _labels(old, new, _ILP_LINKS_CACHE[cache_key], p)

    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    n_o, n_n = S.shape
    cand = set()
    for i in range(n_o):
        for j in np.argsort(-S[i])[: p.topk]:
            cand.add((i, int(j)))
    for j in range(n_n):
        for i in np.argsort(-S[:, j])[: p.topk]:
            cand.add((int(i), j))
    cand = sorted(cand)
    nv = len(cand)
    nvar = nv + n_n + n_o
    c = np.concatenate(
        [[-(S[i, j] - p.tau) for i, j in cand], np.full(n_n, p.gamma_merge), np.full(n_o, p.gamma_split)]
    )
    A = lil_matrix((2 * (n_o + n_n), nvar))
    for k, (i, j) in enumerate(cand):
        A[i, k] = 1
        A[n_o + j, k] = 1
        A[n_o + n_n + i, k] = 1
        A[2 * n_o + n_n + j, k] = 1
    for i in range(n_o):
        A[n_o + n_n + i, nv + n_n + i] = -1
    for j in range(n_n):
        A[2 * n_o + n_n + j, nv + j] = -1
    ub = np.array([p.cap_old] * n_o + [p.cap_new] * n_n + [1] * (n_o + n_n))
    integ = np.concatenate([np.ones(nv), np.zeros(n_n + n_o)])
    res = milp(
        c,
        constraints=[LinearConstraint(A.tocsr(), -np.inf, ub)],
        integrality=integ,
        bounds=Bounds(0, np.concatenate([np.ones(nv), np.full(n_n + n_o, np.inf)])),
    )
    if res.x is not None:
        res.x = res.x[:nv]
    x = np.round(res.x).astype(int) if res.x is not None else np.zeros(nv, int)
    links: dict[int, list[int]] = {}
    for k, (i, j) in enumerate(cand):
        if x[k]:
            links.setdefault(i, []).append(j)
    for i, js in links.items():
        js.sort(key=lambda j: -S[i, j])
        links[i] = [js[0]] + [j for j in js[1:] if S[i, j] >= p.tau + p.second_link_margin]
    _ILP_LINKS_CACHE[cache_key] = links
    return _labels(old, new, links, p)
