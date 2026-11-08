"""Unit and schema-integrity tests for the ReconRAG package and public benchmark."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from reconrag import (
    Params,
    ReconRAGBench,
    char_ratio,
    global_ilp,
    judge,
    layer3,
    parser,
    per_provision,
    sim_matrix,
    substantive_diff,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def test_layer1_article_extraction():
    text = (
        "Pasal 1\n"
        "(1) Dalam Peraturan Menteri ini yang dimaksud dengan Pajak adalah kontribusi wajib.\n"
        "(2) Tarif pajak ditetapkan setiap tahun.\n"
        "Pasal 2\n"
        "Tarif pajak ditetapkan sebesar 10% (sepuluh persen).\n"
    )
    arts = parser.split_articles(text)
    assert len(arts) == 2
    assert arts[0].number == "1"
    assert "1" in arts[0].clauses and "2" in arts[0].clauses
    assert arts[1].number == "2"
    assert "10%" in arts[1].text


def test_layer1_amendment_instructions_and_consolidation():
    base = [
        {"number": "1", "text": "Ketentuan umum tetap berlaku."},
        {"number": "2", "text": "Tarif pajak ditetapkan sebesar 10%."},
        {"number": "5", "text": "Ketentuan sanksi lama."},
    ]
    amend_text = (
        "\nPasal I\n"
        "Beberapa ketentuan dalam Peraturan Menteri diubah sebagai berikut:\n"
        "1. Ketentuan Pasal 2 diubah sehingga berbunyi sebagai berikut:\n"
        "Pasal 2\n"
        "Tarif pajak ditetapkan sebesar 11%.\n"
        "2. Pasal 5 dihapus.\n"
        "\nPasal II\n"
        "Peraturan Menteri ini mulai berlaku pada tanggal diundangkan.\n"
    )
    instrs = parser.parse_instructions(amend_text)
    gold = parser.gold_after_amendment(["1", "2", "5"], instrs)
    assert gold == {"1": "unchanged", "2": "amended", "5": "deleted"}

    consolidated = parser.consolidate(base, amend_text)
    by_no = {c["number"]: c for c in consolidated}
    assert set(by_no.keys()) == {"1", "2"}
    assert "11%" in by_no["2"]["text"]


def test_layer2_global_ilp_and_diff():
    old = [
        {"number": "1", "text": "Ketentuan umum tetap berlaku sama persis untuk seluruh perguruan tinggi negeri."},
        {"number": "2", "text": "Tarif lama ditetapkan sebesar 5 persen dari nilai transaksi penelitian."},
        {"number": "3", "text": "Sanksi administratif khusus bagi pelanggaran pelaporan keuangan yang telah dicabut."},
    ]
    new = [
        {"number": "1", "text": "Ketentuan umum tetap berlaku sama persis untuk seluruh perguruan tinggi negeri."},
        {"number": "2", "text": "Tarif baru ditetapkan sebesar 12 persen dari nilai transaksi penelitian."},
    ]
    S = sim_matrix(old, new)
    prm = Params(tau=0.25, label_mode="diff")
    res1 = global_ilp(old, new, S, prm)
    res2 = global_ilp(old, new, S, prm)
    assert res1.label == res2.label == {"1": "unchanged", "2": "amended", "3": "deleted"}
    assert res1.links["1"] == ["1"]
    assert res1.links["2"] == ["2"]
    assert res1.links["3"] == []


def test_layer2_editorial_vs_substantive_diff():
    a = "Sebagaimana dimaksud pada ayat (1) dan ayat (2) Peraturan Menteri."
    b = "Sebagaimana dimaksud pada ayat (2) dan ayat (3) Peraturan Menteri."
    assert not substantive_diff(a, b)
    assert substantive_diff(a, "Tarif denda dinaikkan menjadi 50 juta rupiah.")
    assert char_ratio(a, a) == 1.0


def test_layer3_policy_and_judge_parser():
    pred = {"P#1": ("amended", {"1"}), "P#2": ("amended", {"2"})}
    decisions = {"P#1": "editorial", "P#2": "different_provision"}
    out, calls, esc = layer3.apply(pred, decisions)
    assert calls == 2 and esc == 0
    assert out["P#1"][0] == "unchanged"
    assert out["P#2"][0] == "deleted"
    assert judge.parse('{"label": "correct"}') == "correct"
    assert judge.parse('{"nilai": "usang"}') == "stale"


def test_data_english_schema_integrity():
    assert (DATA / "regulations_metadata.csv").exists()
    with (DATA / "regulations_metadata.csv").open(encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    assert len(meta) == 57
    assert {r["subset"] for r in meta} == {"amendment", "replacement"}

    amend_rows = [
        json.loads(line)
        for line in (DATA / "amendment_provisions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(amend_rows) == 1199
    assert {r["gold_status"] for r in amend_rows} == {"unchanged", "amended", "deleted"}

    repl_rows = [
        json.loads(line)
        for line in (DATA / "replacement_provisions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(repl_rows) == 444
    assert {r["silver_consensus_status"] for r in repl_rows} <= {
        "unchanged",
        "non_substantive",
        "amended",
        "deleted",
        None,
    }

    q_rows = [
        json.loads(line)
        for line in (DATA / "questions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(q_rows) == 418
    assert {r["question_type"] for r in q_rows} == {"changed", "unchanged", "historical", "deleted"}

    ans_rows = [
        json.loads(line)
        for line in (DATA / "model_answers.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(ans_rows) == 5852
    assert {r["token_rule_label"] for r in ans_rows} <= {"correct", "stale", "mixed", "abstain", "other"}


def test_reconrag_bench_and_context():
    bench = ReconRAGBench(k=5)
    assert len(bench.chunks) == 1457
    assert len(bench.fams) == 47
    ctx = bench.context(
        "Dalam Pedoman Pembentukan Komite Penilaian, aturan mekanisme pelaksanaan penelitian mengacu pada apa?",
        "ReconRAG",
    )
    assert len(ctx) == 5
    assert all(c["status"] in ("in_force", "superseded", "deleted") for c in ctx)
