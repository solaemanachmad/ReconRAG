"""Evaluation of ReconRAG on the released benchmark.

Each module exposes ``run(...) -> list[dict]`` returning table rows; ``report`` writes them to ``results/``.
Run everything with ``python main.py`` from the repository root.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
