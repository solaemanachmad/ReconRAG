"""ReconRAG: reproduce every table and figure of the paper from the released benchmark.

    python main.py                    # all tables and figures -> results/
    python main.py --dense            # also Dense and hybrid retrieval (pip install -e ".[dense]")
    python main.py --only answers     # one part: amendments, replacements, retrieval, answers, judge, figures
    python main.py --check            # recompute and compare with results/ without overwriting (used in CI)
    python main.py --demo             # small example of ingest-time reconciliation

Output (results/):
    amendments.csv    replacements.csv    retrieval.csv    answers.csv
    significance.csv  judge.csv           summary.md       figures/*.pdf
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

PARTS = ["amendments", "replacements", "retrieval", "answers", "judge", "figures"]


def demo() -> None:
    from reconrag.align import Params, global_ilp, sim_matrix

    old = [{"number": "1", "text": "Ketentuan umum mengenai definisi perguruan tinggi."},
           {"number": "2", "text": "Masa studi program sarjana paling lama 7 (tujuh) tahun akademik."},
           {"number": "3", "text": "Ketentuan mengenai evaluasi keberhasilan studi mahasiswa."}]
    new = [{"number": "1", "text": "Ketentuan umum mengenai definisi perguruan tinggi."},
           {"number": "2", "text": "Masa studi program sarjana paling lama 5 (lima) tahun akademik."},
           {"number": "4", "text": "Ketentuan peralihan bagi mahasiswa yang sedang menempuh studi."}]
    res = global_ilp(old, new, sim_matrix(old, new), Params(tau=0.15, label_mode="diff", gamma_merge=0.5, gamma_split=0.5))
    print("Old article -> status (successor)")
    for p in old:
        links = res.links.get(p["number"], [])
        print(f"  {p['number']}: {res.label[p['number']]}" + (f" -> new article {', '.join(links)}" if links else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dense", action="store_true", help="include Dense and hybrid retrieval")
    ap.add_argument("--only", choices=PARTS, nargs="+", help="run only these parts")
    ap.add_argument("--check", action="store_true", help="compare with the tables in results/ instead of writing them")
    ap.add_argument("--demo", action="store_true", help="show a small reconciliation example and exit")
    a = ap.parse_args()
    if a.demo:
        demo()
        return

    from reconrag.evaluation import RESULTS, report

    parts = a.only or PARTS
    if a.check:
        parts = [p for p in parts if p != "figures"]
        mismatches = []
    t0, tables, figures = time.time(), {}, []
    for part in parts:
        print(f"[{part}] running ...", flush=True)
        new: dict[str, list[dict]] = {}
        if part == "amendments":
            from reconrag.evaluation import amendments
            new["amendments"] = amendments.run()
        elif part == "replacements":
            from reconrag.evaluation import replacements
            new["replacements"] = replacements.run()
        elif part == "retrieval":
            from reconrag.evaluation import retrieval
            new["retrieval"] = retrieval.run(dense=a.dense)
        elif part == "answers":
            from reconrag.evaluation import answers
            new["answers"], new["significance"] = answers.run()
        elif part == "judge":
            from reconrag.evaluation import judge
            new["judge"] = judge.run()
        elif part == "figures":
            from reconrag.evaluation import figures as fig
            figures = fig.run()
            print(f"figures -> results/figures/: {', '.join(figures)}\n")
        for name, rows in new.items():
            if a.check:
                diff = report.compare(name, rows)
                mismatches += diff
                print(f"  {name}.csv: " + ("matches" if not diff else f"{len(diff)} difference(s)"))
                continue
            report.write_csv(name, rows)
            print(f"\n{report.TITLES[name]}  ->  results/{name}.csv\n{report.console_table(rows)}\n")
        tables.update(new)

    if a.check:
        for m in mismatches:
            print("  " + m)
        print("All recomputed tables match results/." if not mismatches else "Results differ from results/.")
        sys.exit(1 if mismatches else 0)
    if a.only is None:
        report.write_summary(tables, figures, time.time() - t0, a.dense)
        print(f"Summary -> {RESULTS.relative_to(ROOT)}/summary.md")
    print(f"Done in {time.time() - t0:.0f} s.")


if __name__ == "__main__":
    main()
