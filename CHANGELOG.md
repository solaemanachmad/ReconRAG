# Changelog

## 1.2.0 (2026-11-12)

### Added
- `data/replacement_verification.csv` and `results/verification.csv`: one author verified 50 consensus labels of the full-replacement set (50/50 confirmed; Wilson 95% interval 0.929-1.000).

## 1.1.0 (2026-11-12)

Every number in the paper is now reproduced exactly by `python main.py --dense`, and `python main.py --check` verifies this in continuous integration.

### Fixed
- **Layer-1 cache.** The cache of ILP solutions was keyed on the memory address of the similarity matrix. After garbage collection an address can be reused, so a later regulation pair with a matrix of the same shape could receive another pair's links. The key is now a hash of the matrix contents. In 1.0.0 this changed six reconciliation decisions of the as-published archive and shifted the retrieval results.
- **Full replacements.** The evaluation now follows the definitions of the paper: four status classes (including `non_rule`), the strict three-model consensus (all three annotators agree; 275 test provisions), one ILP per new regulation, the original parameter grid, and both the safe and the automatic Layer-3 policy. 1.0.0 used three classes and a 2-of-3 majority and reported different numbers.
- **Amendment titles.** Document titles in the retrieval archive are now cut at "Perubahan atas" regardless of letter case, as in the experiments; this affects TF-IDF and BM25 scores.
- **Dense and hybrid retrieval** use the same passage text and candidate fusion as the experiments.
- Figures are drawn from the result tables instead of hard-coded numbers; the label of one silver reference in the replacement figure was corrected.

### Added
- `data/answer_prompts.jsonl`: the 2,926 frozen prompts sent to the generators, with the ids and statuses of the context provisions.
- `results/`: reference output (CSV tables, `summary.md`, figures) and `python main.py --check`.
- `CITATION.cff`, this changelog, and a GitHub Actions workflow.
- Schema notes in `data/README.md` for `layer3_decisions.jsonl` and `judge_validation.csv`, the consensus definition, and known text artifacts.

### Changed
- One entry point, `main.py`; the evaluation code moved from `scripts/` to `src/reconrag/evaluation/`.
- `data/human_judge_validation.csv` was renamed to `data/judge_validation.csv`.
- `matplotlib` is a core dependency (figures are part of the default run).

## 1.0.0

Initial release.
