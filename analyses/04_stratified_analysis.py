#!/usr/bin/env python3
"""
04_stratified_analysis.py
=========================
Analysis 3: Performance by evaluation domain stratified by interrater agreement quartiles.

This script analyzes how performance varies across answers where raters mostly
agree vs those where raters disagree. Uses std of ratings per answer as the
agreement measure.

Output:
- output/tables/04_stratified_analysis.csv
- output/figures/04_stratified_analysis_v[1-4].png
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

from analyses.utils import (
    load_data, create_concatenated_answers_df,
    setup_plotting, save_figure_variants, COLORS, PALETTE,
    DIMENSION_NAMES, EVAL_COLS
)

OUTPUT_DIR = project_root / 'output'
TABLES_DIR = OUTPUT_DIR / 'tables'
FIGURES_DIR = OUTPUT_DIR / 'figures'


def compute_answer_agreement(all_answers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute agreement (mean std across dimensions) per answer.

    Returns DataFrame with one row per unique answer.
    """
    results = []

    for answer in all_answers_df['Answer'].unique():
        answer_df = all_answers_df[all_answers_df['Answer'] == answer]
        question = answer_df['Question'].iloc[0]
        question_id = answer_df['question_id'].iloc[0]
        answer_type = answer_df['answer_type'].iloc[0]
        n_raters = len(answer_df)

        row = {
            'Answer': answer,
            'Question': question,
            'question_id': question_id,
            'answer_type': answer_type,
            'n_raters': n_raters
        }

        # Compute std and mean for each dimension
        stds = []
        for domain in EVAL_COLS:
            data = answer_df[domain].dropna()
            if len(data) >= 2:
                std = data.std()
                mean = data.mean()
            else:
                std = np.nan
                mean = data.mean() if len(data) > 0 else np.nan

            dim_name = domain.replace('Eval ', '')
            row[f'{dim_name}_std'] = std
            row[f'{dim_name}_mean'] = mean
            if not np.isnan(std):
                stds.append(std)

        # Overall agreement score (mean std across dimensions)
        row['mean_std'] = np.mean(stds) if stds else np.nan
        results.append(row)

    return pd.DataFrame(results)


def stratify_by_agreement(answer_stats_df: pd.DataFrame) -> pd.DataFrame:
    """Add agreement quartile labels based on mean_std."""

    df = answer_stats_df.copy()

    # Use rank-based quartiles to handle ties
    valid_std = df['mean_std'].dropna()
    ranks = valid_std.rank(method='first')
    quartile_labels = pd.qcut(ranks, q=4, labels=['Q1 (High Agr)', 'Q2', 'Q3', 'Q4 (Low Agr)'])

    # Map back to original dataframe
    df['agreement_quartile'] = np.nan
    df.loc[valid_std.index, 'agreement_quartile'] = quartile_labels.values

    return df


def calculate_stratified_stats(answer_stats_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean performance per dimension per agreement quartile."""

    results = []

    for quartile in ['Q1 (High Agr)', 'Q2', 'Q3', 'Q4 (Low Agr)']:
        df_q = answer_stats_df[answer_stats_df['agreement_quartile'] == quartile]

        for domain in EVAL_COLS:
            dim_name = domain.replace('Eval ', '')
            mean_col = f'{dim_name}_mean'

            if mean_col in df_q.columns:
                values = df_q[mean_col].dropna()
                results.append({
                    'quartile': quartile,
                    'dimension': dim_name,
                    'mean': values.mean(),
                    'std': values.std(),
                    'n': len(values)
                })

    return pd.DataFrame(results)


def create_figure_v1_grouped_bar(stats_df: pd.DataFrame) -> plt.Figure:
    """Create grouped bar chart (dimension x quartile)."""
    setup_plotting()

    fig, ax = plt.subplots(figsize=(14, 7))

    quartiles = ['Q1 (High Agr)', 'Q2', 'Q3', 'Q4 (Low Agr)']
    dimensions = stats_df['dimension'].unique()
    x = np.arange(len(dimensions))
    width = 0.2

    colors = [COLORS['success'], COLORS['primary'], COLORS['tertiary'], COLORS['quaternary']]

    for i, (quartile, color) in enumerate(zip(quartiles, colors)):
        df_q = stats_df[stats_df['quartile'] == quartile]
        means = [df_q[df_q['dimension'] == d]['mean'].values[0]
                 if len(df_q[df_q['dimension'] == d]) > 0 else 0
                 for d in dimensions]
        ax.bar(x + i * width, means, width, label=quartile, color=color, alpha=0.8)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(dimensions, rotation=45, ha='right')
    ax.set_ylabel('Mean Score (1-5)')
    ax.set_xlabel('Evaluation Dimension')
    ax.set_title('Performance by Dimension and Agreement Quartile\n(Q1=High Agreement, Q4=Low Agreement)')
    ax.legend(title='Agreement Quartile')
    ax.set_ylim(0, 5.5)
    ax.axhline(y=3, color=COLORS['neutral'], linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig


def create_figure_v2_heatmap(stats_df: pd.DataFrame) -> plt.Figure:
    """Create heatmap (dimension x quartile)."""
    setup_plotting()

    pivot = stats_df.pivot(index='dimension', columns='quartile', values='mean')
    pivot = pivot[['Q1 (High Agr)', 'Q2', 'Q3', 'Q4 (Low Agr)']]

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        pivot,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        vmin=1,
        vmax=5,
        ax=ax,
        cbar_kws={'label': 'Mean Score'}
    )

    ax.set_title('Performance Heatmap: Dimension × Agreement Quartile')
    ax.set_xlabel('Agreement Quartile (High → Low)')
    ax.set_ylabel('Evaluation Dimension')

    plt.tight_layout()
    return fig


def create_figure_v3_line_plot(stats_df: pd.DataFrame) -> plt.Figure:
    """Create line plot showing trend across quartiles."""
    setup_plotting()

    fig, ax = plt.subplots(figsize=(12, 7))

    quartiles = ['Q1 (High Agr)', 'Q2', 'Q3', 'Q4 (Low Agr)']
    x = np.arange(len(quartiles))
    dimensions = stats_df['dimension'].unique()

    for i, dim in enumerate(dimensions):
        df_dim = stats_df[stats_df['dimension'] == dim]
        means = [df_dim[df_dim['quartile'] == q]['mean'].values[0]
                 if len(df_dim[df_dim['quartile'] == q]) > 0 else np.nan
                 for q in quartiles]
        ax.plot(x, means, 'o-', label=dim, color=PALETTE[i % len(PALETTE)],
                linewidth=2, markersize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(quartiles)
    ax.set_xlabel('Agreement Quartile (High Agreement → Low Agreement)')
    ax.set_ylabel('Mean Score (1-5)')
    ax.set_title('Performance Trend Across Agreement Quartiles')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.set_ylim(1, 5)
    ax.axhline(y=3, color=COLORS['neutral'], linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_figure_v4_small_multiples(answer_stats_df: pd.DataFrame) -> plt.Figure:
    """Create small multiples: one boxplot per dimension."""
    setup_plotting()

    fig, axes = plt.subplots(2, 5, figsize=(16, 8))
    axes = axes.flatten()

    quartiles = ['Q1 (High Agr)', 'Q2', 'Q3', 'Q4 (Low Agr)']
    colors = [COLORS['success'], COLORS['primary'], COLORS['tertiary'], COLORS['quaternary']]

    for i, domain in enumerate(EVAL_COLS):
        ax = axes[i]
        dim_name = domain.replace('Eval ', '')
        mean_col = f'{dim_name}_mean'

        data = [answer_stats_df[answer_stats_df['agreement_quartile'] == q][mean_col].dropna().values
                for q in quartiles]

        bp = ax.boxplot(data, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'], fontsize=8)
        ax.set_title(dim_name[:20], fontsize=9)
        ax.set_ylim(0.5, 5.5)
        ax.axhline(y=3, color=COLORS['neutral'], linestyle='--', alpha=0.5)

        if i % 5 == 0:
            ax.set_ylabel('Score')
        if i >= 5:
            ax.set_xlabel('Quartile')

    plt.suptitle('Score Distribution by Agreement Quartile per Dimension', y=1.02)
    plt.tight_layout()
    return fig


def main():
    """Main function for stratified analysis."""

    print("=" * 60)
    print("04_stratified_analysis.py - Performance by Agreement Quartile")
    print("=" * 60)

    # Create output directories
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n1. Loading data...")
    df = load_data()
    print(f"   Loaded {len(df)} ratings")

    # Create concatenated answers
    print("\n2. Creating concatenated answers dataframe...")
    all_answers_df = create_concatenated_answers_df(df)
    print(f"   Total answer-rater pairs: {len(all_answers_df)}")
    print(f"   Unique answers: {all_answers_df['Answer'].nunique()}")

    # Compute per-answer agreement
    print("\n3. Computing per-answer agreement scores...")
    answer_stats_df = compute_answer_agreement(all_answers_df)
    print(f"   Mean std range: {answer_stats_df['mean_std'].min():.3f} - {answer_stats_df['mean_std'].max():.3f}")

    # Stratify by agreement
    print("\n4. Stratifying by agreement quartiles...")
    answer_stats_df = stratify_by_agreement(answer_stats_df)
    print("   Quartile distribution:")
    print(answer_stats_df['agreement_quartile'].value_counts().sort_index().to_string())

    # Calculate stratified statistics
    print("\n5. Calculating stratified statistics...")
    stats_df = calculate_stratified_stats(answer_stats_df)

    # Save tables
    print("\n6. Saving tables...")
    stats_df.to_csv(TABLES_DIR / '04_stratified_analysis.csv', index=False)
    answer_stats_df.to_csv(TABLES_DIR / '04_stratified_answer_stats.csv', index=False)
    print(f"   Saved to: {TABLES_DIR}")

    # Create figures
    print("\n7. Creating figures...")

    fig1 = create_figure_v1_grouped_bar(stats_df)
    save_figure_variants(fig1, '04_stratified_analysis', FIGURES_DIR, 1)
    print("   - Saved: Grouped bar chart (v1)")

    fig2 = create_figure_v2_heatmap(stats_df)
    save_figure_variants(fig2, '04_stratified_analysis', FIGURES_DIR, 2)
    print("   - Saved: Heatmap (v2)")

    fig3 = create_figure_v3_line_plot(stats_df)
    save_figure_variants(fig3, '04_stratified_analysis', FIGURES_DIR, 3)
    print("   - Saved: Line plot (v3)")

    fig4 = create_figure_v4_small_multiples(answer_stats_df)
    save_figure_variants(fig4, '04_stratified_analysis', FIGURES_DIR, 4)
    print("   - Saved: Small multiples (v4)")

    # Summary
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

    # Find dimensions with biggest difference between Q1 and Q4
    pivot = stats_df.pivot(index='dimension', columns='quartile', values='mean')
    diff = pivot['Q1 (High Agr)'] - pivot['Q4 (Low Agr)']

    print(f"\nKey findings:")
    print(f"  - Dimension with largest Q1-Q4 difference: {diff.idxmax()} ({diff.max():.2f})")
    print(f"  - Dimension with smallest Q1-Q4 difference: {diff.idxmin()} ({diff.min():.2f})")


if __name__ == '__main__':
    main()
