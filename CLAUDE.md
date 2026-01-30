# Agent Instructions

Goal: Evaluate performance of the Meditron LLM on ICU questions reviewed by experts

# Data
- Source: `/mnt/data1/klug/datasets/meditron/Meditron_ICU.xlsx`
- 8 ICU specialists created 200 questions
- the Meditron LLM generated two answers per question 
- the experts than voted on the best answer ("Vote")
- the best answer (or both if equal == vote=12) were rated by the human expert
- 10 evaluation dimensions (1-5 Likert scale): Alignment with Guidelines, Question Comprehension, Logical Reasoning, Relevance & Completeness, Harmlessness, Fairness, Contextual Awareness, Rater Confidence, Model Confidence, Communication & Clarity
- The expert sometimes suggest improved answers: "First Answer Improved", "Second Answer Improved"

# Analysis Plan
0. Overall performance: performance in each evaluated dimension for all answers
1. Interrater agreement for preferred answer over all questions: how much did experts aggree on the best answer over all questions?
2. Interrater agreement (std) for performance on each evaluation domain for each answer (as done in ./exploration/eval_agreement.ipynb)
3. Performance by evaluation domain stratified by interrater agreement quartiles: what is the performance in questions where raters mostly agree vs those where raters disagree?
4. Correlation of performance in alignement with guidelines and interrater kappa? 
5. Subgroup analysis I: performance over domains by ICU subspeciality 
6. Subgroup analysis II: performance by task type (diagnosis, prognosis, treatment, knowledge retrieval)

# Action plan
- Review litterature on Meditron, required statistical tools, existing LLM evaluations in the ICU
- Read data, create a file to explain what each column represents
- go through the analysis plan step by step and write analysis code
- combine all analyses in a final pdf

# General
- use the meditron conda environment for computation 
- Analyses should be reproducible
- Analyses should be written in python 
- Analyses should be saved to ./analyses
- each analysis should result in a table and 4 options of graphs to represent the data
- output should be saved to ./output
- all output should be combined in a final pdf
- progress should be tracked


