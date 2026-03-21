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

### 4. Circularity in the stratified analysis

**File:** `analyses/04_stratified_analysis.py:64-96`

Agreement quartiles are defined by std of ratings, then performance (mean of those same ratings) is compared across quartiles. On bounded 1-5 Likert scales, std and mean are not independent: extreme means mechanically produce lower std (floor/ceiling effects). The paper's claim that "expert disagreement systematically tracks with lower model performance" (line 103) may be partly artifactual.

**Suggested fix (choose one or combine):**
- (a) Use an external agreement measure (e.g., per-question Gwet's AC1 via irrCAC) to define quartiles, breaking the circularity.
- (b) Add a permutation/simulation null test: generate random Likert ratings with the observed marginal distribution and show that observed Q1-Q4 differences exceed the mechanical baseline.
- (c) At minimum, add a Discussion paragraph acknowledging this limitation. The current Limitations section does not mention it.

---

### 5. No statistical tests for stratified comparisons

**File:** `analyses/04_stratified_analysis.py:105-127`

Q1-Q4 mean differences (deltas 0.31-0.86) are reported as "clinically meaningful" in the paper without any statistical test. The reader cannot distinguish signal from noise.

**Suggested fix:** Add Mann-Whitney U tests (appropriate for ordinal data) comparing Q1 vs Q4 per dimension. Apply Benjamini-Hochberg correction for 10 comparisons. Add effect sizes (Cliff's delta or rank-biserial correlation). Report corrected p-values in Table 3.

---

### 6. Multiple testing not addressed

**Files:** `analyses/05_correlation_analysis.py:188`, `04_stratified_analysis.py`, `06/07_*_analysis.py`

Ten per-dimension correlations (Analysis 05, figure v3), 10 dimension comparisons across 4 quartiles (Analysis 04), and AC1 per dimension per category (Analyses 06-07) are all computed without correction.

**Suggested fix:** Apply Benjamini-Hochberg FDR correction to families of related tests. For exploratory analyses (per-dimension breakdowns), explicitly state they are uncorrected and exploratory.

---

## MODERATE (4 issues) -- Results may be biased or suboptimal

### 7. Nominal weights used for ordinal vote data

**File:** `analyses/02_vote_agreement.py:81`

Votes are remapped to -1, 0, +1 (ordinal), but `CAC(vote_matrix)` defaults to nominal (identity) weights. This penalizes partial disagreements (Vote=1 vs Vote=12) as heavily as full disagreements (Vote=1 vs Vote=2), likely underestimating agreement.

**Suggested fix:** Use `CAC(vote_matrix, weights='ordinal')` or `weights='quadratic'`. Report both nominal and weighted agreement and discuss the difference.

---

### 8. value_domain includes 0 but Likert is 1-5

**File:** `analyses/03_eval_agreement.py:125`

`value_domain=[0, 1, 2, 3, 4, 5]` expands the expected-disagreement denominator for ordinal alpha, deflating the coefficient. Only one outlier of 0 exists in the data.

**Suggested fix:** Secondary to Issue 1 (alpha is degenerate anyway). If alpha is recomputed properly, use `value_domain=[1, 2, 3, 4, 5]` and handle the single 0 outlier (recode to 1 or investigate/exclude).

---

### 9. No clustering/nesting accounted for

**File:** `analyses/05_correlation_analysis.py:96-99`

The same rater contributes to many answers, creating within-rater correlation. `scipy.stats.pearsonr/spearmanr` assumes independence, so standard errors may be too small.

**Suggested fix:** Partially mitigated by answer-level aggregation. For more rigor, use clustered bootstrap CIs (by rater) or a mixed-effects model with rater as random effect (`statsmodels.MixedLM`).

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

### 12. "Novel finding" claim for alignment-agreement correlation

The paper claims (line 133) this correlation is "a novel finding." Given Issue 4 (circularity), part of this correlation is mechanically expected from bounded scales.

**Suggested fix:** Qualify the claim: acknowledge the potential mechanical component and provide simulation evidence (Issue 4b) to demonstrate the effect exceeds the baseline.

---

## Summary

| # | Severity | Issue | Affected Files | Paper Impact |
|---|----------|-------|---------------|-------------|
| 1 | ~~CRITICAL~~ FIXED | ~~Single-item Krippendorff's alpha~~ | `03_eval_agreement.py` | Fixed: per-dimension alpha |
| 2 | ~~CRITICAL~~ FIXED | ~~Inconsistent unit of analysis~~ | `data_loader.py`, all analyses | Fixed: 788 evaluations (was 482) |
| 3 | ~~MAJOR~~ FIXED | ~~Pairwise agreement != kappa~~ | `02_vote_agreement.py` | Fixed: weighted Cohen's kappa |
| 4 | MAJOR | Circularity in stratified analysis | `04_stratified_analysis.py` | Central claim may be artifactual |
| 5 | MAJOR | No statistical tests for Q1-Q4 | `04_stratified_analysis.py` | Table 3 lacks significance tests |
| 6 | MAJOR | Multiple testing uncorrected | Multiple scripts | Inflated false positive risk |
| 7 | MODERATE | Nominal weights for ordinal data | `02_vote_agreement.py` | Agreement underestimated |
| 8 | MODERATE | value_domain includes 0 | `03_eval_agreement.py` | Alpha deflated |
| 9 | MODERATE | No clustering accounted for | `05_correlation_analysis.py` | Standard errors too small |
| 10 | MODERATE | Small subgroup AC1 | `06/07_*_analysis.py` | Unreliable subgroup conclusions |
| 11 | MINOR | Missing data mechanism | Paper draft | Possible selection bias |
| 12 | MINOR | Overclaimed novelty | Paper draft | Weakened by circularity |

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
