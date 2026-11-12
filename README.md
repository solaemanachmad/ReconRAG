# ReconRAG: Ingest-Time Version Reconciliation for Retrieval-Augmented Generation over Evolving Regulations

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24054470.svg)](https://doi.org/10.5281/zenodo.24054470)
[![tests](https://github.com/solaemanachmad/ReconRAG/actions/workflows/tests.yml/badge.svg)](https://github.com/solaemanachmad/ReconRAG/actions/workflows/tests.yml)
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
├── main.py                          # single entry point: reproduces all tables and figures
├── pyproject.toml                   # package metadata (pip install -e .)
├── LICENSE                          # MIT (code); data and annotations under CC-BY 4.0
├── CITATION.cff  CHANGELOG.md
├── .github/workflows/tests.yml      # CI: pytest + python main.py --check
├── .env.example                     # variables for live LLM calls (not needed for reproduction)
├── src/reconrag/                    # Python package
│   ├── align.py                     # layers 1-2: global ILP matching and conservative diff
│   ├── layer3.py                    # layer 3: selective LLM verification and safe escalation
│   ├── rag.py                       # retrieval archive, status labels and successor expansion
│   ├── judge.py                     # answer-judge prompt and token-rule grader
│   ├── llm.py                       # minimal OpenAI-compatible client
│   ├── parser/                      # Indonesian regulation parser and TF-IDF
│   └── evaluation/                  # one module per result table, plus figures and report
├── data/                            # benchmark (English schema; see data/README.md)
│   ├── regulations_metadata.csv     # 57 regulation pairs with source URLs and SHA-256
│   ├── amendment_provisions.jsonl   # 1,199 provisions, gold status and successors
│   ├── replacement_provisions.jsonl # 444 provisions, silver labels from three LLMs
│   ├── questions.jsonl              # 418 questions
│   ├── answer_prompts.jsonl         # 2,926 frozen prompts with context provision ids
│   ├── model_answers.jsonl          # 5,852 answers with judge labels
│   ├── layer3_decisions.jsonl       # cached layer-3 decisions
│   ├── judge_validation.csv         # 53 answers labelled by a human rater
│   ├── replacement_verification.csv # 50 consensus labels verified by one author
│   └── texts/                       # extracted regulation texts, one JSON per regulation
├── results/                         # written by main.py (reference copy included)
│   ├── amendments.csv  replacements.csv  verification.csv  retrieval.csv
│   ├── answers.csv  significance.csv  judge.csv
│   ├── summary.md                   # all tables in one readable file
│   └── figures/                     # paper figures (PDF)
└── tests/                           # unit and schema tests (pytest)
```

---

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add ,dense for Dense and hybrid retrieval: pip install -e ".[dev,dense]"
pytest -q                        # tests
python main.py                   # all tables and figures -> results/   (about 2 minutes)
```

Options:

```bash
python main.py --dense                     # also Dense (multilingual MiniLM) and hybrid RRF retrieval
python main.py --only answers judge        # selected parts: amendments replacements verification retrieval answers judge figures
python main.py --check                     # recompute and compare with results/ (also run in CI)
python main.py --demo                      # small example of ingest-time reconciliation
```

`results/` in the repository holds the reference output of `python main.py --dense`; a fresh run should reproduce every number exactly. Figures are drawn from the CSV tables, so running `--only figures` after editing a table redraws them.

| File | Paper | Content |
|---|---|---|
| `amendments.csv` | Table (Tier 1) | per-provision vs. global matching, character ratio vs. diff, layer 3 |
| `replacements.csv` | Figure (full replacements) | macro-F1 against four silver references, escalation load |
| `verification.csv` | Benchmark section | human verification of 50 consensus labels (agreement, Wilson CI, time) |
| `retrieval.csv` | Table (retrieval) | what reaches the generator, four retrievers |
| `answers.csv` | Tables (answers) and figures | accuracy, stale rate with 95% CI, accuracy per question type, tokens |
| `significance.csv` | Appendix | exact McNemar tests with Holm correction |
| `judge.csv` | Table (rubric) | LLM judge and token rule vs. the human rater |

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

If you use the ReconRAG framework or the benchmark, please cite the paper and the dataset (GitHub's "Cite this repository" button uses [`CITATION.cff`](CITATION.cff)). Changes between versions are listed in [`CHANGELOG.md`](CHANGELOG.md).

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

- **Source Code (`src/`, `tests/`, `main.py`):** MIT License (see [`LICENSE`](LICENSE)).
- **Benchmark Annotations (`data/*.jsonl`, `data/*.csv`):** Creative Commons Attribution 4.0 International (CC-BY 4.0).
- **Statutory Texts (`data/texts/*.json`):** Public domain under Indonesian copyright law (*Undang-Undang Nomor 28 Tahun 2014 tentang Hak Cipta*, Pasal 42 huruf b).
