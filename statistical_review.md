# Statistical Review: Meditron ICU Evaluation Project

**Date:** 2026-03-21
**Scope:** Review of statistical analyses and paper draft for the multi-rater expert evaluation of Meditron-3 on 200 ICU clinical questions.

**Overall assessment:** The project is well-structured with a reproducible pipeline, appropriate choice of agreement metrics (irrCAC), and a thoughtful analysis plan. The issues below are intended to strengthen the work for publication.

---

## CRITICAL (2 issues) -- Results incorrect or uninterpretable

### 1. ~~Krippendorff's alpha computed on a single item~~ FIXED

**File:** `analyses/03_eval_agreement.py`

**Fix applied:** Replaced `compute_agreement_per_answer()` with `compute_std_per_answer()` (per-answer variability) and `compute_alpha_per_dimension()` (proper multi-item alpha using items-by-raters matrix). Summary table now reports one alpha per dimension. Output renamed to `03_eval_agreement_alpha_per_dimension.csv`. Paper draft updated.

---

### 2. ~~Inconsistent unit of analysis across scripts~~ FIXED

**Files:** `analyses/utils/data_loader.py`

**Fix applied:** Investigation revealed the "Eval" columns always rate the PREFERRED answer (not always the first answer). Both `get_rated_answers()` and `create_concatenated_answers_df()` were rewritten:
- Vote=1: FIRST_EVAL_COLS paired with First Answer text
- Vote=2: FIRST_EVAL_COLS paired with Second Answer text (evals rate the preferred=second answer)
- Vote=12: FIRST_EVAL_COLS with First Answer, SECOND_EVAL_COLS with Second Answer

This corrected the sample from 482 to **788 answer evaluations** (was dropping all 306 Vote=2 evaluations). Also removed duplicate local `create_concatenated_answers_df()` from `03_eval_agreement.py`. All analyses (01-07) re-run with updated data. Paper draft updated throughout with corrected numbers.

---

## MAJOR (4 issues) -- Results potentially misleading

### 3. ~~Pairwise "kappa" is actually percent agreement~~ FIXED

**File:** `analyses/02_vote_agreement.py`

**Fix applied:** Replaced raw percent agreement with weighted Cohen's kappa (linear weights, via `sklearn.metrics.cohen_kappa_score`). Linear weights are appropriate for the 3-level ordinal vote scale (-1, 0, +1). Output renamed to `02_vote_agreement_pairwise_kappa.csv`. Also saves number of common questions per pair (`02_vote_agreement_pairwise_n.csv`). Paper draft and figure legend updated. Pairwise kappa now ranges from -0.195 to 0.667 (median 0.318).

---

### 4. ~~Circularity in the stratified analysis~~ FIXED

**File:** `analyses/04_stratified_analysis.py`

**Fix applied:** Added simulation null test (1000 iterations). Ratings drawn from observed marginal distribution per dimension, preserving n_raters per answer. Result: 9/10 dimensions have observed Q1-Q4 deltas significantly exceeding the mechanical null (p < 0.05). Only Model Confidence (p=0.273) not significant. Output: `04_stratified_simulation.csv` and `04_stratified_simulation_v1.png`. Paper draft updated in Results and Discussion.

---

### 5. ~~No statistical tests for stratified comparisons~~ FIXED

**File:** `analyses/04_stratified_analysis.py`

**Fix applied:** Added `test_q1_vs_q4()` function with Mann-Whitney U tests (one-sided), Cliff's delta effect sizes, and Benjamini-Hochberg correction for 10 comparisons. All 10 dimensions significant (p_adj < 0.001) with large effect sizes (Cliff's delta 0.35-0.62). Output: `04_stratified_tests.csv`. Paper Methods and Results updated.

---

### 6. ~~Multiple testing not addressed~~ FIXED

**Files:** `analyses/05_correlation_analysis.py`, `04_stratified_analysis.py`

**Fix applied:**
- Analysis 04: BH correction applied to 10 Mann-Whitney U tests (Issue 5). All significant.
- Analysis 05: Added `compute_per_dimension_correlations()` with Spearman rho and BH correction for 10 per-dimension correlations. 9/10 significant. Output: `05_correlation_per_dimension.csv`. Figure v3 updated with corrected p-values.
- Analyses 06-07: Paper Methods now explicitly states subgroup analyses are exploratory and descriptive.

---

## MODERATE (4 issues) -- Results may be biased or suboptimal

### 7. ~~Nominal weights used for ordinal vote data~~ FIXED

**File:** `analyses/02_vote_agreement.py`

**Fix applied:** Added ordinal-weighted computation via `CAC(vote_matrix, weights='ordinal')`. Both nominal and weighted values are now computed and saved. Weighted values are primary (Fleiss' kappa 0.373, Krippendorff's alpha 0.379, Gwet's AC2 0.342), nominal as sensitivity analysis. Paper Table 2 and Methods updated.

---

### 8. value_domain includes 0 but Likert is 1-5

**File:** `analyses/03_eval_agreement.py:125`

`value_domain=[0, 1, 2, 3, 4, 5]` expands the expected-disagreement denominator for ordinal alpha, deflating the coefficient. Only one outlier of 0 exists in the data.

**Suggested fix:** Secondary to Issue 1 (alpha is degenerate anyway). If alpha is recomputed properly, use `value_domain=[1, 2, 3, 4, 5]` and handle the single 0 outlier (recode to 1 or investigate/exclude).

---

### 9. ~~No clustering/nesting accounted for~~ FIXED

**File:** `analyses/05_correlation_analysis.py`

**Fix applied:** Added `clustered_bootstrap_ci()` (2000 iterations, resampling raters with replacement). Clustered CIs are wider and cross zero (Pearson [-0.450, 0.125], Spearman [-0.470, 0.165]) due to only 8 clusters (raters). Fisher z-transform CIs assuming independence exclude zero. Paper reports both, noting the small number of clusters as a limitation. Output: `05_correlation_analysis.csv` now includes both Fisher and clustered CIs.

---

### 10. Small subgroup AC1 reported without qualification

**Files:** `analyses/06_subspecialty_analysis.py`, `07_task_type_analysis.py`; Paper Tables 4-5

Prognosis (n=4), General surgical (n=14), Other (n=2) are reported with AC1 and interpretation labels despite extremely wide CIs.

**Suggested fix:** Report 95% CIs alongside all AC1 values in Tables 4-5. Add a minimum threshold (e.g., n >= 20) below which AC1 is reported as "N/A - insufficient data" rather than given a Landis-Koch label.

---

## MINOR (2 issues) -- Worth noting

### 11. Missing data mechanism not discussed

~30% of expected second-answer evaluations are missing (by design: only when Vote=12). If Vote=12 is more likely for mediocre answers, this subset is not representative.

**Suggested fix:** Compare dimension scores between Vote=12 and Vote=1/2 questions in a supplementary table.

---

### 12. ~~"Novel finding" claim for alignment-agreement correlation~~ ADDRESSED

The paper claims this correlation is "a novel finding." The circularity concern (Issue 4) has been addressed via simulation null test, confirming the effect is real beyond mechanical baseline. The Discussion now acknowledges the potential mechanical component and cites the simulation evidence.

---

## Summary

| # | Severity | Issue | Affected Files | Paper Impact |
|---|----------|-------|---------------|-------------|
| 1 | ~~CRITICAL~~ FIXED | ~~Single-item Krippendorff's alpha~~ | `03_eval_agreement.py` | Fixed: per-dimension alpha |
| 2 | ~~CRITICAL~~ FIXED | ~~Inconsistent unit of analysis~~ | `data_loader.py`, all analyses | Fixed: 788 evaluations (was 482) |
| 3 | ~~MAJOR~~ FIXED | ~~Pairwise agreement != kappa~~ | `02_vote_agreement.py` | Fixed: weighted Cohen's kappa |
| 4 | ~~MAJOR~~ FIXED | ~~Circularity in stratified analysis~~ | `04_stratified_analysis.py` | Simulation confirms 9/10 real |
| 5 | ~~MAJOR~~ FIXED | ~~No statistical tests for Q1-Q4~~ | `04_stratified_analysis.py` | Fixed: Mann-Whitney + Cliff's delta |
| 6 | ~~MAJOR~~ FIXED | ~~Multiple testing uncorrected~~ | Multiple scripts | BH correction + exploratory label |
| 7 | ~~MODERATE~~ FIXED | ~~Nominal weights for ordinal data~~ | `02_vote_agreement.py` | Fixed: ordinal + nominal reported |
| 8 | MODERATE | value_domain includes 0 | `03_eval_agreement.py` | Alpha deflated |
| 9 | ~~MODERATE~~ FIXED | ~~No clustering accounted for~~ | `05_correlation_analysis.py` | Clustered bootstrap added |
| 10 | MODERATE | Small subgroup AC1 | `06/07_*_analysis.py` | Unreliable subgroup conclusions |
| 11 | MINOR | Missing data mechanism | Paper draft | Possible selection bias |
| 12 | ~~MINOR~~ ADDRESSED | ~~Overclaimed novelty~~ | Paper draft | Simulation supports claim |

## Recommended Fix Priority

| Priority | Issue | Effort | Impact |
|----------|-------|--------|--------|
| 1 | Fix `create_concatenated_answers_df()` to respect Vote (#2) | Low | Affects 5 scripts |
| 2 | Remove per-answer alpha, add per-dimension alpha (#1) | Medium | Fixes Table S2 |
| 3 | Replace percent agreement with Cohen's kappa OR fix text (#3) | Low | Fixes paper/code mismatch |
| 4 | Add Mann-Whitney tests + FDR correction (#5, #6) | Medium | Adds statistical rigor |
| 5 | Add circularity discussion + optional simulation (#4) | Medium | Strengthens central claim |
| 6 | Add ordinal weights to irrCAC (#7) | Low | Potentially higher agreement |
| 7 | Report CIs for subgroup AC1, add min-n threshold (#10) | Low | Honest reporting |
| 8 | Paper text corrections (#3 text, #12 novelty claim) | Low | Accuracy |
