# ReconRAG: Ingest-Time Version Reconciliation for Retrieval-Augmented Generation over Evolving Regulations

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24054470.svg)](https://doi.org/10.5281/zenodo.24054470)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Official code and public benchmark repository for **ReconRAG**, a three-layer ingest-time version reconciliation framework and versioned regulatory RAG benchmark over **57 Indonesian regulation pairs** (47 partial-amendment pairs with 1,199 provisions, 10 full-replacement pairs with 444 provisions) and **418 benchmark questions**.

---

## Overview

When statutes and ministerial regulations evolve through partial amendments (*perubahan*) or full-repeal replacements (*pencabutan*), standard Retrieval-Augmented Generation (RAG) indexes both superseded and current provisions side-by-side without explicit validity links. As a result, high-similarity lexical or dense retrieval frequently surfaces **stale law**, while naive recency filtering destroys **unchanged articles** that only exist in the original parent regulation.

**ReconRAG** resolves version conflicts once at **ingest time** via a three-layer cascade:
1. **Layer 1 (Globally constrained matching):** One integer linear program (`scipy.optimize.milp`) over TF-IDF cosine similarities links old and new provisions jointly for a whole regulation pair, without using provision numbers (merges and splits are allowed but penalised).
2. **Layer 2 (Conservative change detection):** A word-level diff labels a linked pair as unchanged only if every changed token is editorial (numbers, enumerators, roman numerals, short connectives, watermarks). Any other change makes the provision amended. On the 826 test provisions of the implicit amendment setting, Layers 1 and 2 reach macro-F1 0.846 and accuracy 0.990.
3. **Layer 3 (Selective LLM verification):** An LLM (`gemini-3.5-flash-lite`, temperature 0) sees only pairs flagged as amended and may confirm the amendment. Under the *safe* escalation policy, any proposal to flip a label is escalated to a human curator.

At query time, **ReconRAG** tags every retrieved provision with an explicit validity label (in force, or superseded or deleted by a named amending regulation) and automatically expands superseded hits with their linked successor provisions.

---

## Repository Structure

```text
ReconRAG/
├── pyproject.toml                   # Python package metadata (pip install -e .)
├── README.md                        # Repository guide & quickstart
├── LICENSE                          # MIT License (code) & CC-BY 4.0 (annotations)
├── .env.example                     # Example environment variables for live LLM calls
├── src/
│   └── reconrag/                    # Core Python package
│       ├── __init__.py
│       ├── align.py                 # Layers 1-2: global ILP matching & conservative diff
│       ├── layer3.py                # Layer 3: selective LLM verifier & safe escalation policy
│       ├── rag.py                   # ReconRAGBench loader, status tagging & successor expansion
│       ├── judge.py                 # Independent LLM judge & deterministic token-rule grader
│       ├── llm.py                   # OpenAI-compatible API client
│       └── parser/                  # Indonesian statutory parser & TF-IDF vectorizer
│           ├── articles.py
│           ├── amendments.py
│           ├── consolidate.py
│           ├── extract.py
│           └── tfidf.py
├── scripts/                         # Reproducibility scripts organized by task
│   ├── reproduce.py                 # Master one-command reproduction of all paper tables
│   ├── reconcile/
│   │   ├── amendments.py            # Tier 1 evaluation on 47 amendment pairs 
│   │   └── replacements.py          # Tier 1 evaluation on 10 replacement pairs 
│   ├── rag/
│   │   ├── retrieval.py             # Multi-backend retrieval: TF-IDF, BM25, MiniLM, RRF 
│   │   ├── evaluate_qa.py           # End-to-end QA, 95% bootstrap CIs & McNemar tests 
│   │   └── validate_judge.py        # Human validation of the LLM judge (kappa = 0.74)
│   └── figures/
│       └── plot_all.py              # Regenerates all vector PDF figures 
├── data/                            # Standardized English-schema public benchmark
│   ├── README.md                    # Detailed field-by-field dataset schema documentation
│   ├── regulations_metadata.csv     # 57 regulation pairs (47 amendments + 10 replacements)
│   ├── amendment_provisions.jsonl   # 1,199 provisions with gold status & consolidation text
│   ├── replacement_provisions.jsonl # 444 provisions with 3-judge + consensus silver labels
│   ├── questions.jsonl              # 418 questions (changed, unchanged, historical, deleted)
│   ├── model_answers.jsonl          # 5,852 model outputs (2 LLMs x 7 RAG configs x 418 questions)
│   ├── layer3_decisions.jsonl       # cached Layer 3 verifier decisions
│   ├── human_judge_validation.csv   # 53 human-audited model answers for judge calibration
│   └── texts/                       # extracted regulation texts (one JSON per regulation)
├── tests/
│   └── test_reconrag.py             # Unit & schema integrity test suite (pytest)
└── paper/                           # Output directory for regenerated vector PDF figures
    └── figures/
```

---

## Quickstart & One-Command Reproduction

### 1. Install Dependencies

```bash
pip install -e ".[dev]"
# Optional: install sentence-transformers to re-run Dense MiniLM & Hybrid RRF embeddings from scratch
pip install -e ".[dense]"
```

### 2. Run the Test Suite

```bash
pytest -v
```

### 3. Reproduce All Paper Tables & Statistical Tests

You can reproduce all empirical results in the paper offline in under 70 seconds using either `main.py` or `scripts/reproduce.py`:

```bash
# Unified CLI entrypoint (runs reproduction suite by default)
python main.py

# Or run live interactive demo on sample evolving provisions
python main.py --demo

# Or run individual modules via CLI flags:
python main.py --reconcile       # Tier 1: Provision-level reconciliation
python main.py --retrieval       # Tier 2: Multi-backend retrieval 
python main.py --qa              # Tier 2: End-to-end QA & McNemar tests
python main.py --validate-judge  # Calibration against 53 human audits (kappa = 0.74)
python main.py --figures         # Regenerate all vector PDF figures
```

Or call individual evaluation scripts directly:

```bash
# Tier 1: Provision-level reconciliation
python scripts/reconcile/amendments.py
python scripts/reconcile/replacements.py

# Tier 2: Multi-backend retrieval 
python scripts/rag/retrieval.py          # Fast lexical (TF-IDF + BM25)
python scripts/rag/retrieval.py --dense  # Full 4-backend (+ Dense MiniLM & Hybrid RRF)

# Tier 2: End-to-end QA, 95% regulation-pair-clustered bootstrap CIs, and Holm-Bonferroni McNemar tests
python scripts/rag/evaluate_qa.py

# Judge validation against 53 human-annotated outputs, Cohen's kappa = 0.74
python scripts/rag/validate_judge.py

# Regenerate vector PDF figures into paper/figures/
python scripts/figures/plot_all.py
```

---

## Scope of the Release

The repository reproduces every table and figure of the paper **offline from cached model outputs** in `data/` (no API key needed). It does **not** include the scripts that called live LLM APIs to (i) generate the 418 questions, (ii) produce the three-model silver annotations, (iii) generate the 5,852 answers, or (iv) run the LLM judge. The verbatim prompts are given in Appendix B of the paper and the resulting outputs are released as data. `src/reconrag/llm.py` is a minimal OpenAI-compatible client provided for re-running these steps; `.env.example` lists the variables it expects. Re-running them with different model versions will not reproduce the cached numbers exactly.

Note on terminology: the paper uses *regulation pair* for the unit that older code and data columns call `family_id`; the field name is kept for backward compatibility.

---

## Standardized English Benchmark Schema

To maximize international reproducibility while preserving authentic Indonesian statutory wording, all files in `data/` use **standardized English column names, JSON keys, and categorical labels**:

| Task / File | Categorical Field | Standardized English Values |
| :--- | :--- | :--- |
| `regulations_metadata.csv` | `relation_type` | `"amendment"`, `"replacement"` |
| `*_provisions.jsonl` | `gold_status` | `"unchanged"`, `"amended"`, `"deleted"` |
| `layer3_decisions.jsonl` | `llm_decision` | `"substantive"`, `"editorial"`, `"different_provision"`, `"unsure"` |
| `questions.jsonl` | `question_type` | `"changed"`, `"unchanged"`, `"historical"`, `"deleted"` |
| `model_answers.jsonl` | `llm_judge_label`, `token_rule_label` | `"correct"`, `"stale"`, `"mixed"`, `"abstain"`, `"other"` |

See [`data/README.md`](data/README.md) for the complete field-level specification.

---

## Citation

If you use the ReconRAG framework, reconciliation cascade, or benchmark dataset in your research, please cite our paper and dataset:

```bibtex
@article{solaeman2026reconrag,
  title={ReconRAG: Ingest-Time Version Reconciliation for Retrieval-Augmented Generation over Evolving Regulations},
  author={Solaeman, Achmad and Wibirama, Sunu},
  journal={arXiv preprint arXiv:2611.XXXXX},
  year={2026},
  url={https://github.com/solaemanachmad/ReconRAG}
}

@misc{solaeman2026dataset,
  author={Solaeman, Achmad and Wibirama, Sunu},
  title={{ReconRAG}: Benchmark Dataset and Reproducibility Package for Versioned Regulatory {RAG}},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.24054470},
  url={https://doi.org/10.5281/zenodo.24054470}
}
```

---

## Authors & Contact

- **Achmad Solaeman** — Department of Electrical Engineering and Informatics, Vocational College, Universitas Gadjah Mada, Yogyakarta, Indonesia (`achmad.solaeman@ugm.ac.id`)
- **Sunu Wibirama** — Department of Electrical Engineering and Information Technology, Universitas Gadjah Mada, Yogyakarta, Indonesia (`sunu@ugm.ac.id`)

---

## License

- **Source Code (`src/`, `scripts/`, `tests/`, `main.py`):** MIT License (see [`LICENSE`](LICENSE)).
- **Benchmark Annotations (`data/*.jsonl`, `data/*.csv`):** Creative Commons Attribution 4.0 International (CC-BY 4.0).
- **Statutory Texts (`data/texts/*.json`):** Public domain under Indonesian copyright law (*Undang-Undang Nomor 28 Tahun 2014 tentang Hak Cipta*, Pasal 42 huruf b).
