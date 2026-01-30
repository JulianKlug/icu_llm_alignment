#!/usr/bin/env python3
"""
06_subspecialty_analysis.py
===========================
Analysis 5: Performance over domains by ICU subspecialty.

This script uses NLP to classify questions into ICU subspecialties and compares
performance across these categories using concatenated answer-level data.

Output:
- output/tables/06_subspecialty_analysis.csv
- output/figures/06_subspecialty_analysis_v[1-4].png
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
    load_data, create_concatenated_answers_df, classify_subspecialty,
    setup_plotting, save_figure_variants, COLORS, PALETTE, EVAL_COLS
)

OUTPUT_DIR = project_root / 'output'
TABLES_DIR = OUTPUT_DIR / 'tables'
FIGURES_DIR = OUTPUT_DIR / 'figures'


def classify_answers(all_answers_df: pd.DataFrame) -> pd.DataFrame:
    """Classify all answers by subspecialty based on their questions."""

    df = all_answers_df.copy()

    # Classify unique questions
    unique_questions = df[['Question']].drop_duplicates()
    unique_questions['subspecialty'] = unique_questions['Question'].apply(classify_subspecialty)

    # Map back to all answers
    subspecialty_map = unique_questions.set_index('Question')['subspecialty'].to_dict()
    df['subspecialty'] = df['Question'].map(subspecialty_map)

    return df


def calculate_subspecialty_stats(all_answers_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean scores per dimension per subspecialty."""

    results = []

    for subspecialty in all_answers_df['subspecialty'].unique():
        df_sub = all_answers_df[all_answers_df['subspecialty'] == subspecialty]
        n_answers = df_sub['Answer'].nunique()
        n_ratings = len(df_sub)

        for domain in EVAL_COLS:
            dim_name = domain.replace('Eval ', '')
            values = df_sub[domain].dropna()

            if len(values) > 0:
                results.append({
                    'subspecialty': subspecialty,
                    'dimension': dim_name,
                    'mean': values.mean(),
                    'std': values.std(),
                    'n_ratings': len(values),
                    'n_answers': n_answers
                })

    return pd.DataFrame(results)


def get_subspecialty_distribution(all_answers_df: pd.DataFrame) -> pd.Series:
    """Get distribution of subspecialties by unique answers."""
    return all_answers_df.groupby('subspecialty')['Answer'].nunique().sort_values(ascending=False)


def create_figure_v1_grouped_bar(stats_df: pd.DataFrame) -> plt.Figure:
    """Create grouped bar chart."""
    setup_plotting()

    # Get top subspecialties
    subspecialties = stats_df.groupby('subspecialty')['n_ratings'].sum().nlargest(6).index.tolist()
    dimensions = stats_df['dimension'].unique()

    fig, ax = plt.subplots(figsize=(16, 8))

    x = np.arange(len(dimensions))
    width = 0.12

    for i, subspec in enumerate(subspecialties):
        df_sub = stats_df[stats_df['subspecialty'] == subspec]
        means = [df_sub[df_sub['dimension'] == d]['mean'].values[0]
                 if len(df_sub[df_sub['dimension'] == d]) > 0 else 0
                 for d in dimensions]
        ax.bar(x + i * width, means, width, label=subspec, color=PALETTE[i], alpha=0.8)

    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(dimensions, rotation=45, ha='right')
    ax.set_ylabel('Mean Score (1-5)')
    ax.set_xlabel('Evaluation Dimension')
    ax.set_title('Performance by Dimension and Subspecialty')
    ax.legend(title='Subspecialty', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_ylim(0, 5.5)
    ax.axhline(y=3, color=COLORS['neutral'], linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig


def create_figure_v2_heatmap(stats_df: pd.DataFrame) -> plt.Figure:
    """Create heatmap of subspecialty x dimension."""
    setup_plotting()

    pivot = stats_df.pivot(index='subspecialty', columns='dimension', values='mean')

    # Sort by overall mean
    pivot['overall_mean'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('overall_mean', ascending=False).drop('overall_mean', axis=1)

    fig, ax = plt.subplots(figsize=(14, 10))

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

    ax.set_title('Performance Heatmap: Subspecialty × Dimension')
    ax.set_xlabel('Evaluation Dimension')
    ax.set_ylabel('Subspecialty')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    return fig


def create_figure_v3_radar_charts(stats_df: pd.DataFrame) -> plt.Figure:
    """Create radar charts for top subspecialties."""
    setup_plotting()

    top_subspecialties = stats_df.groupby('subspecialty')['n_ratings'].sum().nlargest(4).index.tolist()
    dimensions = stats_df['dimension'].unique()

    fig, axes = plt.subplots(2, 2, figsize=(14, 14), subplot_kw=dict(polar=True))
    axes = axes.flatten()

    n_dims = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]

    for i, subspec in enumerate(top_subspecialties):
        ax = axes[i]
        df_sub = stats_df[stats_df['subspecialty'] == subspec]

        values = [df_sub[df_sub['dimension'] == d]['mean'].values[0]
                  if len(df_sub[df_sub['dimension'] == d]) > 0 else 0
                  for d in dimensions]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, color=PALETTE[i])
        ax.fill(angles, values, alpha=0.25, color=PALETTE[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([d[:12] for d in dimensions], size=8)
        ax.set_ylim(0, 5)
        ax.set_title(subspec, size=12, fontweight='bold', y=1.1)

    plt.suptitle('Performance Profile by Subspecialty', y=1.02)
    plt.tight_layout()
    return fig


def create_figure_v4_boxplots(all_answers_df: pd.DataFrame) -> plt.Figure:
    """Create faceted boxplots by subspecialty."""
    setup_plotting()

    top_subspecialties = all_answers_df.groupby('subspecialty')['Answer'].nunique().nlargest(6).index.tolist()
    df_filtered = all_answers_df[all_answers_df['subspecialty'].isin(top_subspecialties)]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, subspec in enumerate(top_subspecialties):
        ax = axes[i]
        df_sub = df_filtered[df_filtered['subspecialty'] == subspec]

        data = [df_sub[col].dropna().values for col in EVAL_COLS]
        bp = ax.boxplot(data, patch_artist=True)

        for j, patch in enumerate(bp['boxes']):
            patch.set_facecolor(PALETTE[j % len(PALETTE)])
            patch.set_alpha(0.7)

        dim_labels = [col.replace('Eval ', '')[:8] for col in EVAL_COLS]
        ax.set_xticklabels(dim_labels, rotation=45, ha='right', fontsize=7)
        n_answers = df_sub['Answer'].nunique()
        ax.set_title(f'{subspec} (n={n_answers})', fontsize=10)
        ax.set_ylim(0.5, 5.5)
        ax.axhline(y=3, color=COLORS['neutral'], linestyle='--', alpha=0.5)

        if i % 3 == 0:
            ax.set_ylabel('Score')

    plt.suptitle('Score Distribution by Subspecialty', y=1.02)
    plt.tight_layout()
    return fig


def main():
    """Main function for subspecialty analysis."""

    print("=" * 60)
    print("06_subspecialty_analysis.py - Performance by Subspecialty")
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
    print(f"   Unique answers: {all_answers_df['Answer'].nunique()}")

    # Classify answers
    print("\n3. Classifying answers by subspecialty...")
    all_answers_df = classify_answers(all_answers_df)

    subspecialty_dist = get_subspecialty_distribution(all_answers_df)
    print("\n   Subspecialty distribution (unique answers):")
    for subspec, count in subspecialty_dist.items():
        print(f"   - {subspec}: {count}")

    # Calculate statistics
    print("\n4. Calculating subspecialty statistics...")
    stats_df = calculate_subspecialty_stats(all_answers_df)

    # Save tables
    print("\n5. Saving tables...")
    stats_df.to_csv(TABLES_DIR / '06_subspecialty_analysis.csv', index=False)

    classification_df = all_answers_df[['Answer', 'Question', 'subspecialty']].drop_duplicates()
    classification_df.to_csv(TABLES_DIR / '06_subspecialty_classification.csv', index=False)
    print(f"   Saved to: {TABLES_DIR}")

    # Create figures
    print("\n6. Creating figures...")

    fig1 = create_figure_v1_grouped_bar(stats_df)
    save_figure_variants(fig1, '06_subspecialty_analysis', FIGURES_DIR, 1)
    print("   - Saved: Grouped bar chart (v1)")

    fig2 = create_figure_v2_heatmap(stats_df)
    save_figure_variants(fig2, '06_subspecialty_analysis', FIGURES_DIR, 2)
    print("   - Saved: Heatmap (v2)")

    fig3 = create_figure_v3_radar_charts(stats_df)
    save_figure_variants(fig3, '06_subspecialty_analysis', FIGURES_DIR, 3)
    print("   - Saved: Radar charts (v3)")

    fig4 = create_figure_v4_boxplots(all_answers_df)
    save_figure_variants(fig4, '06_subspecialty_analysis', FIGURES_DIR, 4)
    print("   - Saved: Boxplots (v4)")

    # Summary
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

    overall_means = stats_df.groupby('subspecialty')['mean'].mean()
    best_subspec = overall_means.idxmax()
    worst_subspec = overall_means.idxmin()

    print(f"\nKey findings:")
    print(f"  - {len(subspecialty_dist)} subspecialties identified")
    print(f"  - Best performing: {best_subspec} (mean: {overall_means[best_subspec]:.2f})")
    print(f"  - Worst performing: {worst_subspec} (mean: {overall_means[worst_subspec]:.2f})")


if __name__ == '__main__':
    main()
