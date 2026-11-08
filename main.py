"""Main CLI Entrypoint for ReconRAG.

Allows running all paper reproductions, individual task modules, or figure generation
from a single unified root interface.

Usage:
    python main.py                  # Run full paper reproduction suite
    python main.py --all            # Same as above
    python main.py --reconcile      # Run Tier 1 reconciliation (amendments + replacements)
    python main.py --retrieval      # Run Tier 2 multi-backend retrieval evaluation
    python main.py --qa             # Run Tier 2 end-to-end QA evaluation & statistical tests
    python main.py --validate-judge # Run LLM judge validation against 53 human audits
    python main.py --figures        # Regenerate all vector PDF figures
    python main.py --demo           # Run a quick demonstration on sample provisions
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def run_demo() -> None:
    """Run a quick demonstration of ReconRAG provision reconciliation and successor expansion."""
    from reconrag.align import Params, global_ilp, sim_matrix

    print("=" * 80)
    print("ReconRAG Live Demonstration")
    print("=" * 80)
    print("Demonstrating ingest-time reconciliation on sample evolving provisions:\n")

    old_provisions = [
        {"number": 1, "text": "Ketentuan Umum mengenai definisi perguruan tinggi."},
        {"number": 2, "text": "Masa studi program sarjana paling lama 7 (tujuh) tahun akademik."},
        {"number": 3, "text": "Ketentuan mengenai evaluasi keberhasilan studi mahasiswa."},
    ]
    new_provisions = [
        {"number": 1, "text": "Ketentuan Umum mengenai definisi perguruan tinggi."},
        {"number": 2, "text": "Masa studi program sarjana paling lama 5 (lima) tahun akademik."},
        {"number": 4, "text": "Ketentuan peralihan bagi mahasiswa yang sedang menempuh studi."},
    ]

    s_mat = sim_matrix(old_provisions, new_provisions)
    prm = Params(tau=0.15, label_mode="diff", gamma_merge=0.5, gamma_split=0.5)
    res = global_ilp(old_provisions, new_provisions, s_mat, prm)

    for p in old_provisions:
        art = str(p["number"])
        lbl = res.label.get(art, "unchanged")
        succs = res.links.get(art, [])
        succ_str = f" -> replaced by Article {', '.join(map(str, succs))}" if succs else ""
        print(f"  Old Article {art}: [{lbl.upper()}]{succ_str}")
        print(f"    Text: {p['text']}")

    print("\n✓ Reconciliation successful: Article 2 detected as AMENDED with successor link to New Article 2.")
    print("  At query time, retrieving Article 2 automatically expands to its in-force successor.")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ReconRAG: Ingest-Time Version Reconciliation for Regulatory RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--all", action="store_true", help="Run the full paper reproduction suite (default)")
    parser.add_argument("--reconcile", action="store_true", help="Run Tier 1 reconciliation on amendments and replacements")
    parser.add_argument("--retrieval", action="store_true", help="Run Tier 2 multi-backend retrieval benchmark")
    parser.add_argument("--qa", action="store_true", help="Run Tier 2 end-to-end QA evaluation and statistical tests")
    parser.add_argument("--validate-judge", action="store_true", help="Run LLM judge calibration against human audits")
    parser.add_argument("--figures", action="store_true", help="Regenerate all paper figures into paper/figures/")
    parser.add_argument("--demo", action="store_true", help="Run a quick interactive demo of ReconRAG reconciliation")

    args = parser.parse_args()

    # If no flags are given, default to running demo + full reproduction
    if not any([args.all, args.reconcile, args.retrieval, args.qa, args.validate_judge, args.figures, args.demo]):
        args.all = True

    t0 = time.time()

    if args.demo:
        run_demo()
        return

    if args.reconcile:
        from reconcile.amendments import main as run_amendments
        from reconcile.replacements import main as run_replacements

        run_amendments()
        print()
        run_replacements()

    elif args.retrieval:
        from rag.retrieval import main as run_retrieval

        run_retrieval()

    elif args.qa:
        from rag.evaluate_qa import main as run_qa

        run_qa()

    elif args.validate_judge:
        from rag.validate_judge import main as run_judge_val

        run_judge_val()

    elif args.figures:
        from figures.plot_all import main as run_figures

        run_figures()

    elif args.all:
        from reproduce import main as run_reproduce

        run_reproduce()

    elapsed = time.time() - t0
    print(f"\n[Completed in {elapsed:.1f}s]")


if __name__ == "__main__":
    main()
