#!/usr/bin/env python3
"""
generate_report.py
==================
Generate final PDF report combining all analyses.

This script compiles all analysis outputs into a comprehensive PDF report.

Output:
- output/report.pdf
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from analyses.utils import setup_plotting, COLORS

OUTPUT_DIR = project_root / 'output'
TABLES_DIR = OUTPUT_DIR / 'tables'
FIGURES_DIR = OUTPUT_DIR / 'figures'


def add_title_page(pdf: PdfPages):
    """Add title page to the report."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.7, 'Meditron ICU LLM Evaluation', fontsize=28, ha='center', fontweight='bold')
    fig.text(0.5, 0.6, 'Performance Analysis Report', fontsize=20, ha='center')
    fig.text(0.5, 0.45, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', fontsize=12, ha='center')
    fig.text(0.5, 0.35, 'Analysis of 200 ICU Questions Rated by 8 Expert Raters', fontsize=14, ha='center')
    fig.text(0.5, 0.25, '10 Evaluation Dimensions on 1-5 Likert Scale', fontsize=12, ha='center')
    pdf.savefig(fig)
    plt.close(fig)


def add_toc_page(pdf: PdfPages):
    """Add table of contents page."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.9, 'Table of Contents', fontsize=20, ha='center', fontweight='bold')

    toc_items = [
        '1. Executive Summary',
        '2. Data Overview',
        '3. Analysis 0: Overall Performance',
        '4. Analysis 1: Interrater Agreement (Votes)',
        '5. Analysis 2: Interrater Agreement (Evaluation Dimensions)',
        '6. Analysis 3: Performance by Agreement Quartile',
        '7. Analysis 4: Alignment vs Agreement Correlation',
        '8. Analysis 5: Performance by Subspecialty',
        '9. Analysis 6: Performance by Task Type',
        '10. Analysis 7: Cross-specialty Comparison',
        '11. Analysis 8: Clinical Error Analysis',
        '12. Conclusions'
    ]

    for i, item in enumerate(toc_items):
        fig.text(0.15, 0.8 - i * 0.06, item, fontsize=12, ha='left')

    pdf.savefig(fig)
    plt.close(fig)


def add_text_page(pdf: PdfPages, title: str, content: list):
    """Add a text page with title and content."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.1, 0.92, title, fontsize=16, fontweight='bold')

    y_pos = 0.85
    for line in content:
        if line.startswith('##'):
            fig.text(0.1, y_pos, line[2:].strip(), fontsize=13, fontweight='bold')
            y_pos -= 0.04
        elif line.startswith('-'):
            fig.text(0.12, y_pos, line, fontsize=10)
            y_pos -= 0.03
        else:
            fig.text(0.1, y_pos, line, fontsize=11)
            y_pos -= 0.035

        if y_pos < 0.1:
            break

    pdf.savefig(fig)
    plt.close(fig)


def add_figure_page(pdf: PdfPages, title: str, figure_path: Path, description: str = ''):
    """Add a page with a figure."""
    if not figure_path.exists():
        print(f"   Warning: Figure not found: {figure_path}")
        return

    fig = plt.figure(figsize=(11, 8.5))

    # Title
    fig.text(0.5, 0.95, title, fontsize=14, ha='center', fontweight='bold')

    # Load and display figure
    img = plt.imread(figure_path)
    ax = fig.add_axes([0.05, 0.15, 0.9, 0.75])
    ax.imshow(img)
    ax.axis('off')

    # Description
    if description:
        fig.text(0.5, 0.05, description, fontsize=10, ha='center', style='italic')

    pdf.savefig(fig)
    plt.close(fig)


def add_table_page(pdf: PdfPages, title: str, table_path: Path, max_rows: int = 30):
    """Add a page with a table."""
    if not table_path.exists():
        print(f"   Warning: Table not found: {table_path}")
        return

    df = pd.read_csv(table_path)

    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.95, title, fontsize=14, ha='center', fontweight='bold')

    ax = fig.add_axes([0.05, 0.1, 0.9, 0.8])
    ax.axis('off')

    # Truncate if too long
    if len(df) > max_rows:
        df = df.head(max_rows)

    # Create table
    table = ax.table(
        cellText=df.round(3).values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Style header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor(COLORS['light'])

    pdf.savefig(fig)
    plt.close(fig)


def generate_executive_summary() -> list:
    """Generate executive summary content."""

    content = [
        '## Study Overview',
        'This report presents the evaluation of Meditron LLM performance on 200 ICU clinical questions.',
        'Eight ICU specialists rated LLM-generated answers across 10 evaluation dimensions.',
        '',
        '## Key Findings',
        '',
    ]

    # Load key results
    try:
        perf_df = pd.read_csv(TABLES_DIR / '01_overall_performance.csv')
        best_dim = perf_df.loc[perf_df['mean'].idxmax()]
        worst_dim = perf_df.loc[perf_df['mean'].idxmin()]
        content.append(f'- Best dimension: {best_dim["dimension"]} (mean: {best_dim["mean"]:.2f})')
        content.append(f'- Worst dimension: {worst_dim["dimension"]} (mean: {worst_dim["mean"]:.2f})')
        content.append(f'- Overall mean: {perf_df["mean"].mean():.2f}')
    except:
        pass

    try:
        vote_df = pd.read_csv(TABLES_DIR / '02_vote_agreement.csv')
        kappa = vote_df[vote_df['metric'] == "Fleiss' Kappa"]['value'].values[0]
        content.append(f'- Interrater agreement (votes): Fleiss κ = {kappa:.3f}')
    except:
        pass

    content.extend([
        '',
        '## Recommendations',
        '- Focus improvement efforts on lower-scoring dimensions',
        '- Consider task type-specific model fine-tuning',
        '- Address areas of low interrater agreement for clearer guidelines'
    ])

    return content


def main():
    """Main function to generate PDF report."""

    print("=" * 60)
    print("generate_report.py - Generating Final PDF Report")
    print("=" * 60)

    setup_plotting()

    # Create PDF
    report_path = OUTPUT_DIR / 'report.pdf'
    print(f"\nGenerating report: {report_path}")

    with PdfPages(report_path) as pdf:

        # Title page
        print("\n1. Adding title page...")
        add_title_page(pdf)

        # Table of contents
        print("2. Adding table of contents...")
        add_toc_page(pdf)

        # Executive summary
        print("3. Adding executive summary...")
        summary_content = generate_executive_summary()
        add_text_page(pdf, '1. Executive Summary', summary_content)

        # Data overview
        print("4. Adding data overview...")
        data_content = [
            '## Dataset',
            '- Source: Meditron ICU evaluation dataset',
            '- 200 unique clinical questions',
            '- 658 total ratings',
            '- 8 expert raters',
            '',
            '## Evaluation Dimensions',
            '- Alignment with Guidelines',
            '- Question Comprehension',
            '- Logical Reasoning',
            '- Relevance & Completeness',
            '- Harmlessness',
            '- Fairness',
            '- Contextual Awareness',
            '- Rater Confidence',
            '- Model Confidence',
            '- Communication & Clarity',
            '',
            '## Vote Distribution',
            '- Vote 1 (First answer): 222',
            '- Vote 2 (Second answer): 306',
            '- Vote 12 (Both equal): 130'
        ]
        add_text_page(pdf, '2. Data Overview', data_content)

        # Analysis 0: Overall Performance
        print("5. Adding Analysis 0: Overall Performance...")
        add_table_page(pdf, 'Analysis 0: Overall Performance - Summary Statistics',
                       TABLES_DIR / '01_overall_performance.csv')
        for v in range(1, 5):
            fig_path = FIGURES_DIR / f'01_overall_performance_v{v}.png'
            if fig_path.exists():
                add_figure_page(pdf, f'Analysis 0: Overall Performance (Figure {v})', fig_path)

        # Analysis 1: Vote Agreement
        print("6. Adding Analysis 1: Vote Agreement...")
        add_table_page(pdf, 'Analysis 1: Interrater Agreement (Votes)',
                       TABLES_DIR / '02_vote_agreement.csv')
        for v in range(1, 5):
            fig_path = FIGURES_DIR / f'02_vote_agreement_v{v}.png'
            if fig_path.exists():
                add_figure_page(pdf, f'Analysis 1: Vote Agreement (Figure {v})', fig_path)

        # Analysis 2: Eval Agreement
        print("7. Adding Analysis 2: Evaluation Agreement...")
        add_table_page(pdf, 'Analysis 2: Interrater Agreement (Dimensions)',
                       TABLES_DIR / '03_eval_agreement.csv')
        for v in range(1, 5):
            fig_path = FIGURES_DIR / f'03_eval_agreement_v{v}.png'
            if fig_path.exists():
                add_figure_page(pdf, f'Analysis 2: Evaluation Agreement (Figure {v})', fig_path)

        # Analysis 3: Stratified Analysis
        print("8. Adding Analysis 3: Stratified Analysis...")
        add_table_page(pdf, 'Analysis 3: Performance by Agreement Quartile',
                       TABLES_DIR / '04_stratified_analysis.csv')
        for v in range(1, 5):
            fig_path = FIGURES_DIR / f'04_stratified_analysis_v{v}.png'
            if fig_path.exists():
                add_figure_page(pdf, f'Analysis 3: Stratified Analysis (Figure {v})', fig_path)

        # Analysis 4: Correlation
        print("9. Adding Analysis 4: Correlation Analysis...")
        add_table_page(pdf, 'Analysis 4: Alignment vs Agreement Correlation',
                       TABLES_DIR / '05_correlation_analysis.csv')
        for v in range(1, 5):
            fig_path = FIGURES_DIR / f'05_correlation_analysis_v{v}.png'
            if fig_path.exists():
                add_figure_page(pdf, f'Analysis 4: Correlation Analysis (Figure {v})', fig_path)

        # Analysis 5: Subspecialty
        print("10. Adding Analysis 5: Subspecialty Analysis...")
        add_table_page(pdf, 'Analysis 5: Performance by Subspecialty',
                       TABLES_DIR / '06_subspecialty_analysis.csv')
        for v in range(1, 5):
            fig_path = FIGURES_DIR / f'06_subspecialty_analysis_v{v}.png'
            if fig_path.exists():
                add_figure_page(pdf, f'Analysis 5: Subspecialty Analysis (Figure {v})', fig_path)

        # Analysis 6: Task Type
        print("11. Adding Analysis 6: Task Type Analysis...")
        add_table_page(pdf, 'Analysis 6: Performance by Task Type',
                       TABLES_DIR / '07_task_type_analysis.csv')
        for v in range(1, 5):
            fig_path = FIGURES_DIR / f'07_task_type_analysis_v{v}.png'
            if fig_path.exists():
                add_figure_page(pdf, f'Analysis 6: Task Type Analysis (Figure {v})', fig_path)

        # Analysis 7: Cross-specialty Comparison
        print("12. Adding Analysis 7: Cross-specialty Comparison...")
        add_table_page(pdf, 'Analysis 7: ICU vs Nuclear Medicine Comparison',
                       TABLES_DIR / '08_cross_specialty_comparison.csv')
        for v in range(1, 5):
            fig_path = FIGURES_DIR / f'08_cross_specialty_comparison_v{v}.png'
            if fig_path.exists():
                add_figure_page(pdf, f'Analysis 7: Cross-specialty Comparison (Figure {v})', fig_path)

        # Analysis 8: Clinical Error Analysis
        print("13. Adding Analysis 8: Clinical Error Analysis...")
        add_table_page(pdf, 'Analysis 8: Tail Risk Distribution',
                       TABLES_DIR / '09_tail_risk_distribution.csv')
        add_table_page(pdf, 'Analysis 8: Error Category Summary',
                       TABLES_DIR / '09_error_category_summary.csv')
        for v in range(1, 5):
            fig_path = FIGURES_DIR / f'09_clinical_error_analysis_v{v}.png'
            if fig_path.exists():
                add_figure_page(pdf, f'Analysis 8: Clinical Error Analysis (Figure {v})', fig_path)

        # Conclusions
        print("14. Adding conclusions...")
        conclusions = [
            '## Summary',
            'This comprehensive analysis evaluated Meditron LLM performance across',
            'multiple dimensions, task types, and subspecialties.',
            '',
            '## Key Takeaways',
            '- Communication & Clarity scored highest among dimensions',
            '- Relevance & Completeness needs improvement',
            '- Fair interrater agreement on preferred answers',
            '- Performance varies by task type and subspecialty',
            '',
            '## Next Steps',
            '- Review low-scoring dimensions for targeted improvements',
            '- Develop task-specific fine-tuning strategies',
            '- Establish clearer evaluation guidelines for raters',
            '- Conduct follow-up study with improved model versions'
        ]
        add_text_page(pdf, '10. Conclusions', conclusions)

    print("\n" + "=" * 60)
    print("Report generated successfully!")
    print("=" * 60)
    print(f"\nOutput: {report_path}")


if __name__ == '__main__':
    main()
