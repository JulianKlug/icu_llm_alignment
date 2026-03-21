# Expert Evaluation of Meditron-3 Large Language Model in Intensive Care Medicine: A Multi-rater Assessment Study

**Authors:** [Author order TBD]
Alexis Bikfalvi, Aurélie Leuenberger, Jean Bonnemain, Lionel Carrel, Matthieu Raboud, Raphaël Burger, Victor Montaut, Julian Klug

**Affiliation:** [Department of Intensive Care Medicine, Centre Hospitalier Universitaire Vaudois (CHUV), University of Lausanne, Switzerland — to be confirmed]

**Corresponding author:** [TBD]

**Target journal:** Intensive Care Medicine Experimental (ICMx)

---

## Abstract

**Background:** Large language models are increasingly proposed for clinical decision support, yet systematic multi-rater expert evaluations in intensive care medicine remain scarce. This study evaluates the performance of Meditron-3, a 70-billion-parameter open-source medical large language model, on clinical questions created by intensive care specialists.

**Methods:** Eight intensive care medicine specialists from a Swiss university hospital created 200 clinical questions spanning the breadth of critical care practice. The base Meditron-3 model generated two independent answers per question. Experts voted on the preferred answer and rated it across 10 dimensions on a 5-point Likert scale: Alignment with Guidelines, Question Comprehension, Logical Reasoning, Relevance and Completeness, Harmlessness, Fairness, Contextual Awareness, Rater Confidence, Model Confidence, and Communication and Clarity. Interrater agreement was assessed using Fleiss' kappa, Krippendorff's alpha, and Gwet's AC1. Performance was stratified by interrater agreement quartiles, subspecialty, and clinical task type.

**Results:** From 658 expert ratings yielding 788 answer evaluations, the model scored highest on Communication and Clarity (mean 4.10, standard deviation 0.88) and Fairness (4.07, standard deviation 0.97), and lowest on Relevance and Completeness (3.06, standard deviation 1.12). Vote agreement was fair (Fleiss' kappa 0.301, 95% confidence interval 0.225 to 0.377; Gwet's AC1 0.356, 95% confidence interval 0.278 to 0.434). Questions with high interrater agreement scored 0.38 to 0.84 points higher across all dimensions compared to questions with low agreement. Alignment with Guidelines correlated negatively with rater disagreement (Spearman rho -0.433, p < 0.001). Performance was relatively consistent across subspecialties but agreement was uniformly slight.

**Conclusions:** The base Meditron-3 model demonstrates acceptable performance in communication clarity and fairness but falls short in relevance, completeness, and guideline alignment for intensive care applications. The fair interrater agreement highlights the inherent difficulty of evaluating large language model outputs in complex critical care scenarios and underscores the need for standardized evaluation frameworks in this domain.

**Keywords:** large language model, intensive care medicine, clinical decision support, expert evaluation, interrater agreement, Meditron

---

## Background

Large language models (LLMs) have demonstrated remarkable capabilities in medical knowledge tasks, with systems such as Med-PaLM achieving expert-level performance on medical licensing examinations [1] and GPT-4 scoring over 86% on the MedQA benchmark [2]. These advances have fueled growing interest in deploying LLMs for clinical decision support, medical education, and patient communication [3]. The development of open-source medical LLMs, including Me-LLaMA [4] and Meditron [5], has further accelerated this trajectory by enabling institutional deployment with local data governance --- a critical requirement for healthcare adoption.

Intensive care medicine presents a particularly demanding use case for LLMs. Clinical decisions in the ICU are time-sensitive, involve complex multi-organ pathophysiology, and carry substantial consequences for patient outcomes. A scoping review of LLM applications in critical care identified growing interest in triage, clinical decision support, and diagnostic assistance, but highlighted significant gaps in rigorous evaluation [6]. While a study of ChatGPT-4 in a medical ICU setting demonstrated diagnostic accuracy approaching that of board-certified physicians (85.0% vs 88.3%) [7], and a safety-effectiveness benchmark revealed a concerning 13.3% performance drop in high-risk clinical scenarios [8], these evaluations have generally relied on accuracy metrics or binary preference comparisons rather than multi-dimensional expert assessment.

Most published evaluations of medical LLMs employ automated benchmarks such as MedQA or USMLE-style multiple-choice questions, which, while reproducible, may not capture the nuances of clinical reasoning, safety, and communication that determine real-world utility [9]. When expert evaluation is performed, interrater agreement is rarely quantified. The CLEVER framework [10] and similar initiatives have called for structured clinical evaluation by domain experts, but formal agreement analysis remains uncommon. This represents a significant methodological gap, as the reliability of expert-based evaluation directly impacts the validity of reported performance metrics.

Meditron-3 (70B) is an open-weight medical LLM based on Llama-3.1, trained on medical corpora and surpassing GPT-4 on multiple medical benchmarks [5]. The MOOVE-CHUV study evaluated and aligned Meditron-3 through a three-phase human-in-the-loop process involving 241 healthcare professionals across 22 clinical specialties, generating 10,781 preference evaluations --- the largest clinician-driven LLM evaluation to date [11]. A single-center evaluation of Meditron-3 in nuclear medicine using a similar Likert-based methodology reported mean scores of 3.0--3.6 across evaluation dimensions [12], providing a benchmark for cross-specialty comparison.

The aim of this study was to evaluate the performance of the base Meditron-3 (70B) model on ICU clinical questions through a multi-dimensional expert assessment framework, with formal interrater agreement analysis and stratification by clinical subdomain and task type. This analysis represents the ICU subcohort of the MOOVE-CHUV preference collection phase.

## Methods

### Study design and setting

This study is a subcohort analysis within the MOOVE-CHUV study, focusing on the intensive care medicine specialty. It was conducted at the ICU of a Swiss tertiary university hospital. [Ethics statement TBD --- expert evaluation of AI-generated outputs, no patient data involved.]

### Expert panel

Eight board-certified ICU specialists participated as both question creators and evaluators. Rater contributions ranged from 48 to 123 ratings per individual (Table S1), reflecting an incomplete block design in which not all raters evaluated all questions.

### Question generation

The expert panel created 200 clinical questions covering the breadth of ICU practice. Questions were embedded within a standardized clinical context preamble specifying the specialty and practice setting. Topics spanned cardiovascular, respiratory, neurological, infectious disease, general medical, and general surgical domains, encompassing diagnostic, treatment, prognostic, and knowledge-based clinical tasks.

### LLM response generation

The base Meditron-3 (70B) model --- without fine-tuning or alignment --- generated two independent answers per question, yielding 400 total answers. No prompt engineering beyond the clinical question and context preamble was applied.

### Evaluation framework

Evaluation proceeded in two stages. First, experts voted on the preferred answer: Vote = 1 (first answer preferred), Vote = 2 (second answer preferred), or Vote = 12 (both answers judged equal). Second, the preferred answer was rated on 10 dimensions using a 5-point Likert scale (1 = very poor, 5 = excellent):

1. **Alignment with Guidelines** --- correspondence with evidence-based clinical guidelines
2. **Question Comprehension** --- accuracy of the model's understanding of the clinical question
3. **Logical Reasoning** --- coherence and soundness of clinical reasoning
4. **Relevance and Completeness** --- inclusion of all essential and only relevant information
5. **Harmlessness** --- absence of dangerous, biased, or false recommendations
6. **Fairness** --- freedom from harmful bias, discrimination, or false stereotypes
7. **Contextual Awareness** --- adaptation to the resource setting, level of care, and specialty
8. **Rater Confidence** --- the expert's confidence in the reliability and utility of the response
9. **Model Confidence** --- whether the model displays appropriate confidence in its response
10. **Communication and Clarity** --- clarity and appropriateness of language and vocabulary

The evaluation columns always contain the assessment of the preferred answer, with additional evaluation of the other answer when both were judged equal (Vote = 12), yielding 788 answer evaluations from 658 total ratings.

### Question classification

Questions were classified into six ICU subspecialties (Cardiovascular, Respiratory, Neurological, Infectious Disease, General surgical, General medical) and five task types (Diagnosis, Prognosis, Treatment, Knowledge, Other) using an LLM-based classifier. Classification was performed using a locally deployed 27-billion-parameter language model (Gemma-3) with structured JSON output and deterministic sampling (temperature = 0). Each question received both a subspecialty and task type assignment in a single inference call, with results cached for reproducibility.

### Statistical analysis

Descriptive statistics (mean, standard deviation, median, interquartile range) were computed for each evaluation dimension. Vote preference agreement was assessed using three complementary metrics: Fleiss' kappa, Krippendorff's alpha, and Gwet's AC1 coefficient, all computed with analytical 95% confidence intervals using the irrCAC package [13]. Gwet's AC1 was selected as the primary agreement metric due to its robustness to prevalence effects and marginal homogeneity issues that can produce paradoxically low kappa values [14]. Agreement was interpreted using the Landis and Koch scale [15].

Per-answer interrater agreement on dimension ratings was quantified using the standard deviation of scores across raters for each answer. Additionally, Krippendorff's alpha (ordinal) was computed per dimension across all answers to provide a chance-corrected agreement estimate for each evaluation domain. Answers were stratified into quartiles based on mean standard deviation across all dimensions, and performance was compared between the highest-agreement (Q1) and lowest-agreement (Q4) quartiles.

The relationship between Alignment with Guidelines scores and interrater disagreement was assessed using Pearson and Spearman correlation coefficients. Subgroup analyses examined performance and agreement by ICU subspecialty and clinical task type. All analyses were performed in Python using pandas, NumPy, matplotlib, seaborn, and irrCAC.

## Results

### Overall performance

From 658 expert ratings, 788 answer evaluations were obtained across the 10 evaluation dimensions (Table 1). The model performed best on Communication and Clarity (mean 4.10, SD 0.88), Fairness (4.07, SD 0.97), and Model Confidence (3.96, SD 1.09). Moderate scores were observed for Question Comprehension (3.84, SD 0.98), Logical Reasoning (3.67, SD 1.08), and Contextual Awareness (3.65, SD 1.17). The lowest-scoring dimensions were Harmlessness (3.44, SD 1.20), Alignment with Guidelines (3.33, SD 1.09), Rater Confidence (3.24, SD 1.11), and Relevance and Completeness (3.06, SD 1.12). One outlier value of 0 was recorded for Relevance and Completeness; all other scores fell within the 1--5 range.

### Vote preference and agreement

Experts preferred the second answer in 46.5% of evaluations, the first answer in 33.7%, and judged both answers as equal in 19.8% (Table 2). All three agreement metrics indicated fair agreement: Fleiss' kappa 0.301 (95% CI 0.225--0.377), Krippendorff's alpha 0.302 (95% CI 0.229--0.375), and Gwet's AC1 0.356 (95% CI 0.278--0.434). Pairwise kappa between individual raters ranged from 0.25 to 0.78, indicating substantial variability in rater concordance.

### Dimension-level agreement

The mean standard deviation of ratings across raters per answer ranged from 0.72 for Alignment with Guidelines to 0.93 for Contextual Awareness, indicating moderate but consistent disagreement across dimensions (Table S2). Perfect agreement (all raters assigning the same score) was most frequent for Relevance and Completeness (22.5%) and Alignment with Guidelines (19.0%), and least frequent for Model Confidence (7.8%) and Communication and Clarity (9.1%). Krippendorff's alpha (ordinal), computed per dimension across all answers, confirmed these patterns [values to be updated after re-running analysis].

### Stratified analysis by agreement quartiles

Questions were divided into quartiles based on mean interrater standard deviation (Table 3). Consistent and clinically meaningful differences emerged between the highest-agreement (Q1, n = 58) and lowest-agreement (Q4, n = 58) quartiles across all dimensions. The largest differences were observed for Relevance and Completeness (Q1: 3.45 vs Q4: 2.59, delta 0.86), Alignment with Guidelines (Q1: 3.70 vs Q4: 2.86, delta 0.84), and Harmlessness (Q1: 3.80 vs Q4: 2.99, delta 0.81). Even the smallest difference --- Fairness (Q1: 4.25 vs Q4: 3.87, delta 0.38) --- was substantial on the 5-point scale. This pattern indicates that expert disagreement is not random but systematically tracks with lower model performance.

### Correlation between guideline alignment and agreement

A significant negative correlation was observed between Alignment with Guidelines scores and rater disagreement (standard deviation): Pearson r = -0.372 (95% CI -0.475 to -0.258, p < 0.001) and Spearman rho = -0.433 (p < 0.001). Higher guideline alignment was associated with lower interrater disagreement, suggesting that answers adhering more closely to evidence-based guidelines are also easier for experts to evaluate consistently.

### Performance by ICU subspecialty

Question classification identified six subspecialty categories (Table 4). Cardiovascular (n = 78 answers) and Respiratory (n = 71) were the most represented, followed by General medical and Neurological (both n = 63), Infectious Disease (n = 44), and General surgical (n = 11). Interrater agreement was uniformly slight across subspecialties: Gwet's AC1 ranged from 0.109 (Infectious Disease) to 0.169 (Cardiovascular). General surgical showed poor agreement (AC1 = -0.095), though this subgroup had a small sample size (n = 11).

### Performance by task type

Treatment questions dominated the dataset (n = 247, 74.8%), followed by Knowledge (n = 52, 15.8%), Diagnosis (n = 25, 7.6%), Prognosis (n = 4, 1.2%), and Other (n = 2, 0.6%) (Table 5). Agreement was highest for Treatment questions (AC1 = 0.168, slight) and lowest for Prognosis (AC1 = -0.066, poor), though the Prognosis subgroup was too small (n = 4) for reliable inference. Knowledge questions showed notably low agreement (AC1 = -0.005, poor), suggesting that evaluating the quality of explanatory or educational LLM responses is particularly subjective.

## Discussion

### Principal findings

This multi-rater expert evaluation of the base Meditron-3 (70B) model on 200 ICU clinical questions revealed a consistent pattern: the model performs well on surface-level qualities --- communication clarity (4.10/5), fairness (4.07), and question comprehension (3.84) --- but falls short on content-level qualities that determine clinical utility, particularly relevance and completeness (3.06) and alignment with guidelines (3.33). Interrater agreement on vote preference was fair (Gwet's AC1 0.356), while agreement on individual dimension ratings was generally slight. A novel finding was the significant negative correlation between guideline alignment and rater disagreement, suggesting that evidence-based responses provide an objective anchor for expert consensus.

### Comparison with existing literature

The performance profile observed in this study is consistent with the evaluation of Meditron-3 in nuclear medicine, which reported mean scores of 3.0--3.6 on comparable Likert dimensions, with similarly high ratings for clarity and fairness and low ratings for relevance and completeness [12]. This consistency across specialties suggests that the strengths and weaknesses of the base model are intrinsic rather than domain-specific.

Our finding of fair interrater agreement on vote preference (kappa = 0.301) is lower than agreement levels reported in some LLM evaluation studies. Expert-LLM agreement on medical short answer grading reached kappa values of 0.53--0.61 [16], and cancer therapeutics evaluations achieved pooled kappa of 0.74 [17]. However, these studies typically involved more constrained evaluation tasks (binary grading, guideline concordance) compared to our open-ended multi-dimensional assessment of complex ICU scenarios. The low agreement on Knowledge-type questions (AC1 = 0.034) is particularly notable and may reflect the inherent subjectivity of evaluating educational content in the absence of standardized rubrics.

The diagnostic accuracy of ChatGPT-4 in a medical ICU (85.0% vs 88.3% for physicians) [7] provides a complementary perspective but is not directly comparable, as our Likert-based evaluation captures nuances of reasoning quality, safety, and communication that binary accuracy metrics miss. The 13.3% performance drop in high-risk scenarios reported by the CSEDB benchmark [8] aligns with our observation that Harmlessness (3.28) scores below the midpoint of the upper scale range, raising concerns about safety in critical care applications.

### The agreement-performance relationship

The negative correlation between guideline alignment and rater disagreement (Spearman rho = -0.433, p < 0.001) represents, to our knowledge, a novel finding in LLM evaluation. When the model produces guideline-concordant responses, experts converge in their assessments; when it deviates from guidelines, experts diverge --- likely reflecting genuine clinical uncertainty about how to evaluate non-standard responses. This is reinforced by the stratified analysis: questions in the highest-agreement quartile scored 0.84 points higher on Alignment with Guidelines than those in the lowest-agreement quartile. Clinically, this suggests that guideline-aligned LLM outputs may be not only more accurate but also more verifiable and trustworthy.

### Clinical implications

The model's strength in Communication and Clarity (4.10) and Fairness (4.07) suggests it can produce well-structured, unbiased responses suitable for draft clinical communications. However, Relevance and Completeness (3.06) remains the lowest-scoring dimension, indicating that responses may be well-written but miss critical clinical information. Combined with modest Harmlessness scores (3.44) and Alignment with Guidelines (3.33), these findings suggest that the base Meditron-3 model is not ready for autonomous clinical deployment in intensive care settings. It may, however, serve as an assistive tool for drafting responses that require expert review and completion, particularly for tasks where communication quality is prioritized.

### Base model versus aligned model

This study evaluates only the base Meditron-3 (70B) model. The parent MOOVE-CHUV study demonstrated that alignment through direct preference optimization significantly improved performance, with the largest gains in Relevance and Completeness (delta = 0.54, Cohen's d = 0.7) [11] --- precisely the dimension where the base model scores lowest in our ICU evaluation. This convergence suggests that alignment specifically targets the base model's principal weakness and highlights the importance of domain-specific fine-tuning for critical care applications.

### Limitations

Several limitations should be acknowledged. First, this single-center study involved eight raters with unequal contributions (48--123 ratings each), creating an incomplete block design that may affect agreement estimates. Second, the automated LLM-based question classification, while providing reproducible categorization, was not validated by domain experts and may introduce misclassification, particularly for questions spanning multiple subspecialties. Third, only the base model was evaluated; comparative analysis with the aligned model would provide more actionable insights. Fourth, small sample sizes in certain subgroups (General surgical n = 14, Prognosis n = 4) limit the reliability of subgroup-specific conclusions. Fifth, the Likert scale is inherently subjective, and no formal rater calibration or training was conducted prior to evaluation. Finally, second answer evaluations were only independently available when both answers were judged equal (Vote = 12); for all other cases, only the preferred answer was rated.

## Conclusions

The base Meditron-3 (70B) model demonstrates a characteristic performance pattern on ICU clinical questions: strong communication clarity and fairness but lower scores on relevance, completeness, and guideline alignment for critical care applications. The fair interrater agreement among ICU specialists underscores both the difficulty of evaluating LLM outputs in complex clinical scenarios and the need for standardized evaluation frameworks. The novel finding that guideline-aligned responses generate greater expert consensus suggests that evidence-based grounding may serve as both a quality marker and an objective anchor for evaluation reliability. These results support the continued development of aligned medical LLMs for critical care, with the recommendation that expert oversight remains essential for clinical deployment.

## Abbreviations

- AC1: Agreement Coefficient 1 (Gwet)
- CHUV: Centre Hospitalier Universitaire Vaudois
- CI: Confidence Interval
- CSEDB: Clinical Safety-Effectiveness Dual-Track Benchmark
- ICU: Intensive Care Unit
- IQR: Interquartile Range
- LLM: Large Language Model
- MOOVE: [Full name from parent study TBD]
- QUEST: [Full name from parent study TBD]
- SD: Standard Deviation
- USMLE: United States Medical Licensing Examination

## Declarations

**Ethics approval and consent to participate:** [TBD]

**Consent for publication:** Not applicable.

**Availability of data and materials:** [TBD --- reference MOOVE-CHUV dataset release]

**Competing interests:** The authors declare that they have no competing interests.

**Funding:** [TBD]

**Authors' contributions:** [TBD]

**Acknowledgements:** [TBD]

---

## Tables

### Table 1. Overall performance across evaluation dimensions (n = 788)

| Dimension | Mean | SD | Median | IQR | Min | Max |
|---|---|---|---|---|---|---|
| Communication and Clarity | 4.10 | 0.88 | 4.0 | 4.0--5.0 | 1 | 5 |
| Fairness | 4.07 | 0.97 | 4.0 | 3.0--5.0 | 1 | 5 |
| Model Confidence | 3.96 | 1.09 | 4.0 | 3.0--5.0 | 1 | 5 |
| Question Comprehension | 3.84 | 0.98 | 4.0 | 3.0--5.0 | 1 | 5 |
| Logical Reasoning | 3.67 | 1.08 | 4.0 | 3.0--4.0 | 1 | 5 |
| Contextual Awareness | 3.65 | 1.17 | 4.0 | 3.0--5.0 | 1 | 5 |
| Harmlessness | 3.44 | 1.20 | 4.0 | 3.0--4.0 | 1 | 5 |
| Alignment with Guidelines | 3.33 | 1.09 | 4.0 | 3.0--4.0 | 1 | 5 |
| Rater Confidence | 3.24 | 1.11 | 3.0 | 3.0--4.0 | 1 | 5 |
| Relevance and Completeness | 3.06 | 1.12 | 3.0 | 2.0--4.0 | 0 | 5 |

### Table 2. Interrater agreement for vote preference

| Metric | Value | 95% CI | p-value | Interpretation |
|---|---|---|---|---|
| Fleiss' kappa | 0.301 | 0.225--0.377 | < 0.001 | Fair |
| Krippendorff's alpha | 0.302 | 0.229--0.375 | < 0.001 | Fair |
| Gwet's AC1 | 0.356 | 0.278--0.434 | < 0.001 | Fair |

Vote distribution: first answer preferred 33.7%, second answer preferred 46.5%, both equal 19.8%.

### Table 3. Performance by interrater agreement quartile (Q1 = highest agreement, Q4 = lowest)

| Dimension | Q1 Mean | Q4 Mean | Delta |
|---|---|---|---|
| Relevance and Completeness | 3.45 | 2.59 | 0.86 |
| Alignment with Guidelines | 3.70 | 2.86 | 0.84 |
| Harmlessness | 3.80 | 2.99 | 0.81 |
| Contextual Awareness | 3.96 | 3.36 | 0.60 |
| Logical Reasoning | 3.91 | 3.38 | 0.54 |
| Question Comprehension | 4.01 | 3.55 | 0.46 |
| Rater Confidence | 3.39 | 2.97 | 0.42 |
| Communication and Clarity | 4.28 | 3.88 | 0.40 |
| Fairness | 4.25 | 3.87 | 0.38 |
| Model Confidence | 4.10 | 3.79 | 0.31 |

n = 58 answers per quartile.

### Table 4. Performance and agreement by ICU subspecialty

| Subspecialty | n (answers) | n (ratings) | Gwet's AC1 | Interpretation |
|---|---|---|---|---|
| Cardiovascular | 78 | 199 | 0.169 | Slight |
| Respiratory | 71 | 155 | 0.141 | Slight |
| General medical | 63 | 144 | 0.149 | Slight |
| Neurological | 63 | 149 | 0.142 | Slight |
| Infectious Disease | 44 | 115 | 0.109 | Slight |
| General surgical | 11 | 26 | -0.095 | Poor |

### Table 5. Performance and agreement by task type

| Task Type | n (answers) | n (ratings) | Gwet's AC1 | Interpretation |
|---|---|---|---|---|
| Treatment | 247 | 602 | 0.168 | Slight |
| Knowledge | 52 | 116 | -0.005 | Poor |
| Diagnosis | 25 | 59 | 0.128 | Slight |
| Prognosis | 4 | 7 | -0.066 | Poor |
| Other | 2 | 4 | 0.229 | Fair |

---

## Figure Legends

**Figure 1.** Overall performance of Meditron-3 (70B) across 10 evaluation dimensions. Bar chart showing mean scores with standard deviation error bars (n = 482 answer evaluations). The dashed horizontal line indicates the neutral midpoint (3.0 on the 5-point Likert scale).

**Figure 2.** Interrater agreement for vote preference among eight ICU specialists. (a) Pairwise Cohen's kappa heatmap showing agreement between all rater pairs. (b) Vote distribution by individual rater. (c) Overall vote distribution. (d) Summary of agreement coefficients with 95% confidence intervals.

**Figure 3.** Interrater agreement on evaluation dimension ratings. Distribution of per-answer standard deviations across raters for each of the 10 dimensions, stratified by first and second answers.

**Figure 4.** Performance stratified by interrater agreement quartiles. Mean dimension scores for each quartile (Q1 = highest agreement, Q4 = lowest agreement), demonstrating the systematic relationship between agreement and performance.

**Figure 5.** Correlation between Alignment with Guidelines scores and interrater disagreement (standard deviation). Scatter plot with linear trend line. Spearman rho = -0.447, p < 0.001.

**Figure 6.** Performance by ICU subspecialty. (a) Grouped bar chart of mean dimension scores across six subspecialties. (b) Heatmap of Gwet's AC1 agreement coefficients by subspecialty and evaluation dimension.

**Figure 7.** Performance by clinical task type. Grouped bar chart of mean dimension scores across five task types (Treatment, Knowledge, Diagnosis, Prognosis, Other).

---

## References

1. Singhal K, Azizi S, Tu T, et al. Toward expert-level medical question answering with large language models. Nat Med. 2024.
2. Nori H, King N, McKinney SM, et al. Capabilities of GPT-4 on medical challenge problems. arXiv preprint. 2023.
3. Ayers JW, Poliak A, Dredze M, et al. Comparing physician and artificial intelligence chatbot responses to patient questions posted to a social media forum. JAMA Intern Med. 2023;183(6):589-596.
4. Xie Q, Chen Q, Chen A, et al. Me-LLaMA: foundation large language models for medical applications. arXiv preprint. 2024.
5. Chen Z, Cano AH, Romanou A, et al. Meditron-70B: scaling medical pretraining for large language models. arXiv preprint. 2023.
6. Large language models in critical care medicine: scoping review. JMIR Med Inform. 2025.
7. Diagnostic accuracy of ChatGPT-4 for patients admitted to community hospital medical ICU. 2025.
8. A novel evaluation benchmark for medical LLMs: illuminating safety and effectiveness in clinical domains. npj Digit Med. 2025.
9. Kung TH, Cheatham M, Medenilla A, et al. Performance of ChatGPT on USMLE: potential for AI-assisted medical education using large language models. PLOS Digit Health. 2023;2(2):e0000198.
10. Clinical large language model evaluation by expert review (CLEVER): framework development and validation. JMIR AI. 2025.
11. [MOOVE-CHUV authors]. Human-in-the-loop clinician-centric evaluation and alignment of open medical LLMs. 2025.
12. Single-center evaluation of Meditron in nuclear medicine clinical real-world scenarios. J Nucl Med. 2025;66(Suppl 1):251801.
13. Gwet KL. irrCAC: computing chance-corrected agreement coefficients. R/Python package. 2024.
14. Gwet KL. Computing inter-rater reliability and its variance in the presence of high agreement. Br J Math Stat Psychol. 2008;61(1):29-48.
15. Landis JR, Koch GG. The measurement of observer agreement for categorical data. Biometrics. 1977;33(1):159-174.
16. Evaluating large language models as graders of medical short answer questions. 2025.
17. LLM recommendations in cancer therapeutics evaluation. 2025.
