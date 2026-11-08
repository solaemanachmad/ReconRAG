"""Master One-Command Reproduction Script for ReconRAG.

Reproduces all core empirical results reported in the paper using only the
clean English-schema benchmark files in ``data/``:
  1. Tier 1 Amendment Reconciliation (47 pairs, 1,199 provisions)
  2. Tier 1 Replacement Reconciliation across 4 Silver References (10 pairs, 444 provisions)
  3. Tier 2 Multi-Backend Retrieval (TF-IDF + BM25, 418 queries over 3,529 provisions)
  4. Tier 2 End-to-End QA + 95% Family-Cluster Bootstrap CIs + Holm-Bonferroni McNemar Tests (5,852 outputs)
  5. Human Validation of the LLM Answer Judge (53 stratified rows, Cohen's kappa = 0.74 — Section 5.4)

Usage:
    python scripts/reproduce.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from reconcile.amendments import main as run_amendments  # noqa: E402
from reconcile.replacements import main as run_replacements  # noqa: E402
from rag.retrieval import main as run_retrieval  # noqa: E402
from rag.evaluate_qa import main as run_qa  # noqa: E402
from rag.validate_judge import main as run_judge_val  # noqa: E402


def main() -> None:
    t0 = time.time()
    print("================================================================================")
    print("ReconRAG — Full Benchmark & Paper Reproduction Suite")
    print("================================================================================\n")

    run_amendments()
    print()
    run_replacements()
    print()
    sys.argv = [sys.argv[0]]  # default fast lexical mode for retrieval
    run_retrieval()
    print()
    run_qa()
    print()
    run_judge_val()

    elapsed = time.time() - t0
    print("\n================================================================================")
    print(f"All ReconRAG evaluations reproduced cleanly in {elapsed:.1f}s.")
    print("================================================================================")


if __name__ == "__main__":
    main()
