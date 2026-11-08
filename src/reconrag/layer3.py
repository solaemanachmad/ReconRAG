"""Layer 3: Selective LLM verification on pairs flagged as 'amended' by Layer 2, with Safe Escalation Policy.

LLM decision per flagged pair:
  - "substantive"         -> remains "amended"
  - "editorial"           -> flipped to "unchanged" (automatic policy) or escalated (safe policy)
  - "different_provision" -> flipped to "deleted"   (automatic policy) or escalated (safe policy)
  - "unsure"              -> remains "amended" and escalated to curator
"""
from __future__ import annotations

import difflib
import json
import re

PROMPT = """Anda memeriksa dua versi ketentuan peraturan menteri di Indonesia. Teks LAMA berasal dari versi sebelumnya;
teks BARU adalah ketentuan yang dipasangkan dengannya di versi berikutnya. PERBEDAAN: [-dihapus-] {{+ditambah+}}.

TEKS LAMA:
{old_text}

TEKS BARU:
{new_text}

PERBEDAAN:
{diff}

Tentukan satu keputusan:
- "substantif": isi aturan berubah sehingga pengguna (mahasiswa, pegawai, masyarakat, instansi) bisa mendapat jawaban
  berbeda untuk pertanyaan yang sama: angka, jangka waktu, pihak/pejabat yang berwenang, syarat, cakupan, kewajiban,
  larangan, atau ayat/huruf berisi ketentuan yang ditambah atau dihilangkan.
- "redaksional": hanya ejaan, tanda baca, penomoran, rujukan pasal yang bergeser, istilah setara, atau nama
  kementerian/pejabat yang berubah karena reorganisasi dengan fungsi yang sama.
- "beda_ketentuan": kedua teks sebenarnya mengatur hal yang berbeda (pasangan ini keliru).
- "ragu": tidak dapat diputuskan dengan yakin.
Balas JSON saja: {{"keputusan": "substantif|redaksional|beda_ketentuan|ragu", "alasan": "satu kalimat"}}"""

_DEC_NORM = {
    "substantif": "substantive",
    "substantive": "substantive",
    "redaksional": "editorial",
    "editorial": "editorial",
    "beda_ketentuan": "different_provision",
    "different_provision": "different_provision",
    "ragu": "unsure",
    "unsure": "unsure",
}


def word_diff(a: str, b: str) -> str:
    x, y = a.split(), b.split()
    k = lambda w: re.sub(r"[^\w]", "", w.lower())
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, [k(w) for w in x], [k(w) for w in y], autojunk=False).get_opcodes():
        if op == "equal":
            out += x[i1:i2]
        else:
            if i2 > i1:
                out.append("[-" + " ".join(x[i1:i2]) + "-]")
            if j2 > j1:
                out.append("{+" + " ".join(y[j1:j2]) + "+}")
    return " ".join(out)


def build_prompt(old_text: str, new_texts: list[str]) -> str:
    new = "\n\n".join(new_texts)
    return PROMPT.format(old_text=old_text[:3500], new_text=new[:3500], diff=word_diff(old_text, new)[:4000])


def parse(raw: str) -> str:
    dec = json.JSONDecoder()
    for i, ch in enumerate(raw or ""):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(raw[i:])
                if isinstance(obj, dict) and obj.get("keputusan") in _DEC_NORM:
                    return _DEC_NORM[obj["keputusan"]]
            except json.JSONDecodeError:
                continue
    return "unsure"


def apply(pred: dict[str, tuple[str, set]], decisions: dict[str, str]) -> tuple[dict, int, int]:
    """Automatic Layer-3 policy: directly apply LLM label flips."""
    out, calls, esc = dict(pred), 0, 0
    for k, raw_dec in decisions.items():
        if k not in out or out[k][0] != "amended":
            continue
        dec = _DEC_NORM.get(raw_dec, raw_dec)
        calls += 1
        if dec == "editorial":
            out[k] = ("unchanged", out[k][1])
        elif dec == "different_provision":
            out[k] = ("deleted", set())
        elif dec == "unsure":
            esc += 1
    return out, calls, esc


def apply_escalate(pred: dict[str, tuple[str, set]], decisions: dict[str, str], gold: dict) -> tuple[dict, int, int]:
    """Safe Escalation Policy: LLM may only confirm 'amended'; any proposal to flip is escalated to a curator."""
    out, calls, esc = dict(pred), 0, 0
    for k, raw_dec in decisions.items():
        if k not in out or out[k][0] != "amended":
            continue
        dec = _DEC_NORM.get(raw_dec, raw_dec)
        calls += 1
        if dec != "substantive":
            esc += 1
            if k in gold:
                out[k] = (gold[k][0], set(gold[k][1]) if len(gold[k]) > 1 else out[k][1])
    return out, calls, esc
