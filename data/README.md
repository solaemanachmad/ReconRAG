# ReconRAG Benchmark: Versioned Regulatory RAG Dataset

Official public benchmark accompanying the paper:
> **ReconRAG: Ingest-Time Version Reconciliation for Retrieval-Augmented Generation over Evolving Regulations**

This repository directory (`data/`) provides the complete, standardized **English-schema** dataset for evaluating (1) **Tier 1: Provision-Level Version Reconciliation** (across explicit amendments, implicit amendments, and full repeal-and-replace regulations) and (2) **Tier 2: End-to-End Version-Aware Retrieval-Augmented Generation (RAG)**.

While the underlying statutory provisions, questions, and answers are preserved in their authentic **Indonesian** legal text to maintain statutory fidelity, **all file names, column headers, JSON keys, domain names, status labels, question categories, and evaluation labels are standardized in English** for out-of-the-box international reproducibility.

---

## 1. Dataset Summary & Statistics

| Component | File | Records | Splits | Description |
|---|---|---:|---|---|
| **Regulation Metadata** | `regulations_metadata.csv` | 57 pairs | `dev`, `test` | 47 amendment pairs + 10 full replacement pairs with official BPK URLs and PDF SHA-256 hashes |
| **Amendment Set (Gold)** | `amendment_provisions.jsonl` | 1,199 provisions | `dev` (373), `test` (826) | Provision-level texts (`old_text`, `in_force_text`), official `gold_status`, and `successor_articles` across `education`, `health`, and `manpower` |
| **Replacement Set (Silver)** | `replacement_provisions.jsonl` | 444 provisions | `dev` (146), `test` (298) | Full repeal-and-replace provisions with independent 3-model silver annotations (`gemini`, `gptoss`, `claude`) and majority consensus |
| **QA Benchmark** | `questions.jsonl` | 418 questions | `dev` (148), `test` (270) | Rule-validated questions covering `changed`, `unchanged`, `historical`, and `deleted` provisions |
| **Model Outputs & Grades** | `model_answers.jsonl` | 5,852 outputs | `dev`, `test` | Full responses from `openai/gpt-oss-120b` and `qwen/qwen3.8-27b` across 7 RAG conditions, graded by both the LLM Judge and Token-Rule Grader |

---

## 2. Schema Reference

### 2.1 `regulations_metadata.csv`
Lists every regulation pair in the benchmark together with provenance links from the Audit Board of Indonesia's Legal Documentation Network ([JDIH BPK](https://peraturan.bpk.go.id)):
- `pair_id`: Unique pair identifier (`<old_regulation_id>__<new_regulation_id>`)
- `subset`: `"amendment"` (47 pairs) or `"replacement"` (10 pairs)
- `family_id`: Regulation family identifier (used for family-stratified splits and bootstrap resampling)
- `old_regulation_id` / `new_regulation_id`: Document identifiers
- `intermediate_amendment_id`: Intermediate amending document ID (for consolidated replacement bases, if any)
- `domain`: `"education"`, `"health"`, or `"manpower"`
- `split`: `"dev"` or `"test"`
- `old_title` / `new_title`: Official regulation titles
- `old_details_url` / `new_details_url`: Landing page URLs on JDIH BPK
- `old_pdf_url` / `new_pdf_url`: Direct PDF download URLs
- `old_pdf_sha256` / `new_pdf_sha256`: SHA-256 checksums of the source PDF files

### 2.2 `amendment_provisions.jsonl` (1,199 provisions)
Each line is a JSON object representing one base provision and its fate after the amending regulation:
```json
{
  "provision_id": "permenristekdikti-44-2015#66",
  "pair_id": "permenristekdikti-44-2015__permenristekdikti-50-2018",
  "base_regulation_id": "permenristekdikti-44-2015",
  "amendment_regulation_id": "permenristekdikti-50-2018",
  "domain": "education",
  "split": "dev",
  "article_number": "66",
  "gold_status": "amended",
  "successor_articles": ["66"],
  "old_text": "...",
  "in_force_text": "..."
}
```
- `gold_status`: Derived deterministically from official Indonesian legislative drafting formulas (*Pasal I* instructions):
  - `"unchanged"` ($n=990$): Provision remains in force without modification.
  - `"amended"` ($n=197$): Provision is restated with substantive or editorial modifications; `successor_articles` includes the amended article and any adjacent inserted articles (e.g., `12A`).
  - `"deleted"` ($n=12$): Provision is explicitly repealed (`dihapus`); `in_force_text` is `null`.

### 2.3 `replacement_provisions.jsonl` (444 provisions)
Each line represents a provision from a regulation that was repealed and replaced in full (without article-by-article mapping instructions):
- `silver_gemini_status`, `silver_gptoss_status`, `silver_claude_status`: Independent status labels (`"unchanged"`, `"amended"`, `"deleted"`, `"non_substantive"`) assigned by three frontier LLMs.
- `silver_*_successors`: List of successor article numbers in `new_regulation_id` identified by each model.
- `silver_consensus_status` & `silver_consensus_successors`: Majority vote ($\ge 2$ of 3 models) and successor intersection.

### 2.4 `questions.jsonl` (418 questions)
Each line is a validated question paired with both the currently valid answer and the superseded answer:
```json
{
  "question_id": "permenristekdikti-27-2019#8#changed",
  "legacy_qid": "permenristekdikti-27-2019#8#berubah",
  "base_regulation_id": "permenristekdikti-69-2016",
  "amendment_regulation_id": "permenristekdikti-27-2019",
  "domain": "education",
  "split": "dev",
  "article_number": "8",
  "question_type": "changed",
  "question": "Dalam Pedoman Pembentukan Komite Penilaian Dan/Atau Reviewer, aturan mengenai mekanisme pelaksanaan penelitian mengacu pada apa?",
  "in_force_answer": "Peraturan Menteri Riset, Teknologi, dan Pendidikan Tinggi",
  "superseded_answer": "ketentuan peraturan perundang-undangan mengenai Pengadaan Barang/Jasa Pemerintah"
}
```
- `question_type`:
  - `"changed"` ($n=109$; 70 test): Asks about the current rule for an amended provision (correct answer = `in_force_answer`; answering with `superseded_answer` counts as `"stale"`).
  - `"unchanged"` ($n=188$; 124 test): Control questions on unamended provisions from the base regulation.
  - `"historical"` ($n=109$; 70 test): Explicitly asks about the rule *prior* to the amendment (correct answer = `superseded_answer`).
  - `"deleted"` ($n=12$; 6 test): Asks about a repealed provision (correct answer = stating that the provision was deleted/repealed or abstaining; answering with `superseded_answer` counts as `"stale"`).

### 2.5 `model_answers.jsonl` (5,852 graded outputs)
Contains every generated answer across both open-weight generators (`openai/gpt-oss-120b` and `qwen/qwen3.8-27b`) and all experimental conditions:
- `condition`:
  - `"Naive_RAG"` (`B0`): Unfiltered archive, no version metadata.
  - `"Recency_Prompt"` (`B1`): Unfiltered archive with document title/year and recency instruction.
  - `"Latest_Doc"` (`B2`): Indexes only the latest document per regulation family.
  - `"Query_Time_Judge"` (`Q`): Two-step query-time conflict detection and resolution.
  - `"ReconRAG"` (`R+succ`): **Proposed method** — ingest-time version reconciliation with status labels and automatic successor expansion.
  - `"ReconRAG_Route"` (`R+route`): Exploratory ablation adding heuristic historical-cue query routing.
  - `"Oracle"` (`O+succ`): Theoretical upper bound using official gold amendment instructions.
- `llm_judge_label` & `token_rule_label`:
  - `"correct"`: Matches the target version for the question type.
  - `"stale"`: Uses the outdated provision (`superseded_answer`) on a current-rule question (`changed` or `deleted`).
  - `"mixed"`: Mentions both versions without resolving which applies.
  - `"abstain"`: Outputs `"TIDAK DITEMUKAN"` (not found).
  - `"other"`: Incorrect or irrelevant answer.

---

## 3. Quick Start (Python)

```python
import json
from collections import Counter

# Load test-split graded outputs for ReconRAG on qwen3.8-27b
rows = [
    json.loads(line)
    for line in open("data/model_answers.jsonl", encoding="utf-8")
]
sub = [
    r for r in rows
    if r["split"] == "test"
    and r["generator_model"] == "qwen/qwen3.8-27b"
    and r["condition"] == "ReconRAG"
]

acc = sum(r["llm_judge_label"] == "correct" for r in sub) / len(sub)
cur = [r for r in sub if r["question_type"] in ("changed", "deleted")]
stale = sum(r["llm_judge_label"] == "stale" for r in cur) / len(cur)
print(f"Total accuracy: {acc:.2%} | Stale rate on current-rule questions: {stale:.2%}")
```

To run all evaluation tables and reproduce the paper results:
```bash
python scripts/reproduce.py
```

---

## 4. License & Legal Provenance
Under Article 42(b) of Indonesia's Copyright Law (*Undang-Undang Nomor 28 Tahun 2014 tentang Hak Cipta*), statutory laws and government regulations (*peraturan perundang-undangan*) are in the public domain (no copyright applies). Our benchmark annotations, successor links, questions, evaluation rubrics, and scripts are released under the **CC-BY 4.0** (data) and **MIT** (code) licenses.
