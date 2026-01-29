# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project evaluates the alignment of Large Language Models (LLMs) for use in Intensive Care Unit (ICU) settings. The focus is on assessing whether LLM outputs are safe, appropriate, and aligned with clinical best practices for critical care environments.

## Commands

Use the `meditron` conda environment for all Python commands:

```bash
# Run full analysis (Table 1, metrics, plots)
/home/klug/utils/miniconda3/envs/meditron/bin/python analysis/meditron_analysis.py

# Run subspecialty analysis
/home/klug/utils/miniconda3/envs/meditron/bin/python analysis/subspecialty_analysis.py

# Generate PDF report (run after both analyses above)
/home/klug/utils/miniconda3/envs/meditron/bin/python analysis/generate_report.py
```

## Project Structure

- `analysis/` - Analysis scripts
  - `meditron_analysis.py` - Main analysis: Table 1, metrics, inter-rater reliability, plots
  - `subspecialty_analysis.py` - ICU subspecialty classification and analysis
  - `subspecialty_classification_method.md` - Documentation of classification algorithm
  - `generate_report.py` - PDF report generator combining all analyses
- `output/` - Generated outputs (CSVs, plots, PDF report)

## Data

- Source: `/mnt/data1/klug/datasets/meditron/Meditron_ICU.xlsx`
- 658 expert ratings from 8 ICU specialists on 200 questions
- 10 evaluation dimensions (1-5 Likert scale): Alignment with Guidelines, Question Comprehension, Logical Reasoning, Relevance & Completeness, Harmlessness, Fairness, Contextual Awareness, Rater Confidence, Model Confidence, Communication & Clarity
- Pre-alignment (658 ratings) and post-alignment (130 paired ratings) evaluations

## Domain Context

- ICU clinical decision support requires high reliability and safety
- Evaluation dimensions reflect clinical accuracy, patient safety, and appropriate uncertainty quantification
- Inter-rater reliability metrics assess consistency across expert evaluators
