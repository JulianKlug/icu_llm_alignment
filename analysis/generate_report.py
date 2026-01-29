#!/usr/bin/env python3
"""
Generate PDF Report for Meditron ICU Evaluation Analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

OUTPUT_DIR = Path('/home/klug/icu_projects/icu_llm_alignment/output')


def add_title_page(pdf):
    """Add a title page to the report."""
    fig = plt.figure(figsize=(11, 8.5))

    fig.text(0.5, 0.65, 'Meditron ICU Evaluation Report',
             fontsize=28, fontweight='bold', ha='center', va='center')

    fig.text(0.5, 0.55, 'Human Expert Assessment of LLM Responses\nto ICU-Related Clinical Questions',
             fontsize=16, ha='center', va='center')

    fig.text(0.5, 0.40, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
             fontsize=12, ha='center', va='center', color='gray')

    fig.text(0.5, 0.25, 'Analysis Summary:\n'
             '• 658 expert ratings on 200 unique ICU questions\n'
             '• 8 clinical expert raters\n'
             '• 10 evaluation dimensions (1-5 Likert scale)\n'
             '• Pre and post-alignment comparison (130 paired ratings)',
             fontsize=11, ha='center', va='center')

    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def add_table_page(pdf, df, title, description=None):
    """Add a table page to the report."""
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis('off')

    # Title
    fig.text(0.5, 0.95, title, fontsize=16, fontweight='bold', ha='center')

    if description:
        fig.text(0.5, 0.90, description, fontsize=10, ha='center', style='italic')

    # Create table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0.05, 0.1, 0.9, 0.75]
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)

    # Header styling
    for j, col in enumerate(df.columns):
        table[(0, j)].set_facecolor('#2c3e50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def add_image_page(pdf, image_path, title):
    """Add an image page to the report."""
    fig = plt.figure(figsize=(11, 8.5))

    fig.text(0.5, 0.97, title, fontsize=14, fontweight='bold', ha='center')

    img = mpimg.imread(image_path)
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.88])
    ax.imshow(img)
    ax.axis('off')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def add_section_header(pdf, title, subtitle=None):
    """Add a section header page."""
    fig = plt.figure(figsize=(11, 8.5))

    fig.text(0.5, 0.55, title, fontsize=32, fontweight='bold', ha='center', va='center')

    if subtitle:
        fig.text(0.5, 0.45, subtitle, fontsize=14, ha='center', va='center', color='gray')

    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def add_summary_page(pdf):
    """Add a summary/conclusions page."""
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis('off')

    fig.text(0.5, 0.92, 'Key Findings Summary', fontsize=18, fontweight='bold', ha='center')

    summary_text = """
DATASET OVERVIEW
• Total of 658 expert ratings collected from 8 ICU specialists
• 200 unique clinical questions evaluated
• Each question rated by 2-25 experts (mean: 3.29 ratings per question)
• 130 paired ratings available for pre/post-alignment comparison

PRE-ALIGNMENT EVALUATION RESULTS
• Overall mean score: 3.74 (SD: 1.06) on a 1-5 scale
• Highest-rated dimensions:
  - Communication & Clarity: 4.19
  - Fairness: 4.16
  - Model Confidence: 4.05
• Lowest-rated dimensions:
  - Relevance & Completeness: 3.19
  - Rater Confidence: 3.30
  - Alignment with Guidelines: 3.43

POST-ALIGNMENT EVALUATION RESULTS (n=130 paired ratings)
• Overall mean score: 3.12 (SD: 1.28)
• Scores showed minimal change from pre-alignment
• No significant improvements observed in any dimension

SUBSPECIALTY ANALYSIS
• Questions classified into 11 ICU subspecialties
• Top categories: Cardiovascular (30%), Sepsis (21%), Respiratory (17.5%)
• Highest scores: Procedures & Monitoring (4.23), Toxicology (4.07)
• No significant difference between subspecialties (Kruskal-Wallis p=0.20)

INTER-RATER RELIABILITY
• All 200 questions had ≥2 raters
• Mean exact agreement: 51.0% across all dimensions
• Mean within-1-point agreement: 84.2% across all dimensions

CONCLUSIONS
• Meditron demonstrates moderate performance on ICU-related questions
• The model excels in communication clarity and perceived fairness
• Alignment process did not significantly improve expert ratings
• Performance is consistent across ICU subspecialties
"""

    fig.text(0.08, 0.85, summary_text, fontsize=9, va='top',
             family='monospace', linespacing=1.4)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def main():
    """Generate the complete PDF report."""

    report_path = OUTPUT_DIR / 'meditron_icu_evaluation_report.pdf'

    with PdfPages(report_path) as pdf:
        print("Generating PDF report...")

        # Title page
        print("  - Title page")
        add_title_page(pdf)

        # Table 1: Descriptive Statistics
        print("  - Table 1: Descriptive Statistics")
        table1 = pd.read_csv(OUTPUT_DIR / 'table1_descriptive_stats.csv')
        add_table_page(pdf, table1, 'Table 1: Dataset Descriptive Statistics',
                      'Overview of the evaluation dataset structure and rater contributions')

        # Evaluation Metrics Table
        print("  - Evaluation Metrics")
        metrics = pd.read_csv(OUTPUT_DIR / 'evaluation_metrics.csv')
        metrics_display = metrics[['Dimension', 'Condition', 'N', 'Mean', 'SD', 'Median']].copy()
        metrics_display['Mean'] = metrics_display['Mean'].round(2)
        metrics_display['SD'] = metrics_display['SD'].round(2)
        add_table_page(pdf, metrics_display, 'Table 2: Evaluation Metrics by Dimension',
                      'Mean scores (1-5 scale) for each evaluation dimension')

        # Paired Comparison Table
        print("  - Paired Comparison")
        paired = pd.read_csv(OUTPUT_DIR / 'paired_comparison.csv')
        add_table_page(pdf, paired, 'Table 3: Pre vs Post-Alignment Comparison (Paired Ratings)',
                      'Statistical comparison of 130 paired ratings before and after alignment')

        # Inter-rater Reliability Table
        print("  - Inter-rater Reliability")
        irr = pd.read_csv(OUTPUT_DIR / 'inter_rater_reliability.csv')
        add_table_page(pdf, irr, 'Table 4: Inter-Rater Reliability',
                      'Agreement metrics between expert raters')

        # Plots
        print("  - Paired comparison plot")
        add_image_page(pdf, OUTPUT_DIR / 'paired_comparison.png',
                      'Figure 1: Pre vs Post-Alignment Evaluation Scores')

        print("  - Dimension distributions plot")
        add_image_page(pdf, OUTPUT_DIR / 'dimension_distributions.png',
                      'Figure 2: Score Distributions by Evaluation Dimension')

        print("  - Score distributions plot")
        add_image_page(pdf, OUTPUT_DIR / 'score_distributions.png',
                      'Figure 3: Overall Score Distributions')

        print("  - Rater heatmap")
        add_image_page(pdf, OUTPUT_DIR / 'rater_heatmap.png',
                      'Figure 4: Mean Scores by Rater and Dimension')

        print("  - Dimension correlation plot")
        add_image_page(pdf, OUTPUT_DIR / 'dimension_correlation.png',
                      'Figure 5: Correlation Between Evaluation Dimensions')

        # Subspecialty Analysis Section
        print("  - Subspecialty section header")
        add_section_header(pdf, 'Subspecialty Analysis',
                          'Performance breakdown by ICU clinical domain')

        # Subspecialty Summary Table
        print("  - Subspecialty summary table")
        subspecialty_summary = pd.read_csv(OUTPUT_DIR / 'subspecialty_summary.csv')
        add_table_page(pdf, subspecialty_summary,
                      'Table 5: Evaluation Scores by ICU Subspecialty',
                      'Questions classified by clinical domain with overall performance metrics')

        # Subspecialty Plots
        print("  - Subspecialty distribution")
        add_image_page(pdf, OUTPUT_DIR / 'subspecialty_distribution.png',
                      'Figure 6: Distribution of Questions by ICU Subspecialty')

        print("  - Subspecialty scores")
        add_image_page(pdf, OUTPUT_DIR / 'subspecialty_scores.png',
                      'Figure 7: Overall Evaluation Scores by ICU Subspecialty')

        print("  - Subspecialty heatmap")
        add_image_page(pdf, OUTPUT_DIR / 'subspecialty_heatmap.png',
                      'Figure 8: Evaluation Dimension Scores by Subspecialty')

        print("  - Subspecialty boxplot")
        add_image_page(pdf, OUTPUT_DIR / 'subspecialty_boxplot.png',
                      'Figure 9: Score Distribution by ICU Subspecialty')

        # Summary page
        print("  - Summary page")
        add_summary_page(pdf)

    print(f"\nReport saved to: {report_path}")
    return report_path


if __name__ == '__main__':
    main()
