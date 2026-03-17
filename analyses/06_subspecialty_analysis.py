#!/usr/bin/env python3
"""
06_subspecialty_analysis.py
===========================
Analysis 5: Performance over domains by ICU subspecialty.

This script uses NLP to classify questions into ICU subspecialties and compares
performance across these categories using concatenated answer-level data.
Uses irrCAC to compute Gwet's AC1 as the primary agreement metric.

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
from irrCAC.raw import CAC

from analyses.utils import (
    load_data, create_concatenated_answers_df, classify_subspecialty,
    setup_plotting, save_figure_variants, COLORS, PALETTE, EVAL_COLS
)

OUTPUT_DIR = project_root / 'output'
TABLES_DIR = OUTPUT_DIR / 'tables'
FIGURES_DIR = OUTPUT_DIR / 'figures'


def interpret_agreement(value: float) -> str:
    """Interpret agreement coefficient using Landis & Koch's guidelines."""
    if value < 0:
        return "Poor"
    elif value < 0.20:
        return "Slight"
    elif value < 0.40:
        return "Fair"
    elif value < 0.60:
        return "Moderate"
    elif value < 0.80:
        return "Substantial"
    else:
        return "Almost Perfect"


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


def compute_agreement_for_dimension(df_subset: pd.DataFrame, domain: str) -> dict:
    """
    Compute Gwet's AC1 for a specific dimension within a subset of data.

    Creates a matrix with answers as rows and raters as columns.
    """
    # Pivot: rows=answers, columns=raters, values=ratings
    try:
        rating_matrix = df_subset.pivot_table(
            index='Answer',
            columns='Name',
            values=domain,
            aggfunc='first'
        )

        # Need at least 2 answers and 2 raters with data
        if rating_matrix.shape[0] < 2 or rating_matrix.shape[1] < 2:
            return {'gwet_ac1': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan, 'n_answers': rating_matrix.shape[0]}

        # Check if there's enough non-null data
        non_null_counts = rating_matrix.notna().sum(axis=1)
        valid_rows = (non_null_counts >= 2).sum()

        if valid_rows < 2:
            return {'gwet_ac1': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan, 'n_answers': rating_matrix.shape[0]}

        cac = CAC(rating_matrix)
        gwet_result = cac.gwet()
        gwet_est = gwet_result['est']

        return {
            'gwet_ac1': gwet_est['coefficient_value'],
            'ci_lower': gwet_est['confidence_interval'][0],
            'ci_upper': gwet_est['confidence_interval'][1],
            'n_answers': rating_matrix.shape[0]
        }
    except Exception as e:
        return {'gwet_ac1': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan, 'n_answers': 0}


def calculate_subspecialty_stats(all_answers_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean scores and Gwet's AC1 per dimension per subspecialty."""

    results = []

    for subspecialty in all_answers_df['subspecialty'].unique():
        df_sub = all_answers_df[all_answers_df['subspecialty'] == subspecialty]
        n_answers = df_sub['Answer'].nunique()
        n_ratings = len(df_sub)

        for domain in EVAL_COLS:
            dim_name = domain.replace('Eval ', '')
            values = df_sub[domain].dropna()

            if len(values) > 0:
                # Compute agreement
                agreement = compute_agreement_for_dimension(df_sub, domain)

                results.append({
                    'subspecialty': subspecialty,
                    'dimension': dim_name,
                    'mean': values.mean(),
                    'std': values.std(),
                    'n_ratings': len(values),
                    'n_answers': n_answers,
                    'gwet_ac1': agreement['gwet_ac1'],
                    'gwet_ci_lower': agreement['ci_lower'],
                    'gwet_ci_upper': agreement['ci_upper']
                })

    return pd.DataFrame(results)


def calculate_overall_agreement_by_subspecialty(all_answers_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate overall Gwet's AC1 per subspecialty (across all dimensions)."""

    results = []

    for subspecialty in all_answers_df['subspecialty'].unique():
        df_sub = all_answers_df[all_answers_df['subspecialty'] == subspecialty]
        n_answers = df_sub['Answer'].nunique()

        # Compute AC1 for each dimension and average
        ac1_values = []
        for domain in EVAL_COLS:
            agreement = compute_agreement_for_dimension(df_sub, domain)
            if not np.isnan(agreement['gwet_ac1']):
                ac1_values.append(agreement['gwet_ac1'])

        mean_ac1 = np.mean(ac1_values) if ac1_values else np.nan

        results.append({
            'subspecialty': subspecialty,
            'n_answers': n_answers,
            'n_ratings': len(df_sub),
            'mean_gwet_ac1': mean_ac1,
            'interpretation': interpret_agreement(mean_ac1) if not np.isnan(mean_ac1) else 'N/A'
        })

    return pd.DataFrame(results).sort_values('n_answers', ascending=False)


def get_subspecialty_distribution(all_answers_df: pd.DataFrame) -> pd.Series:
    """Get distribution of subspecialties by unique answers."""
    return all_answers_df.groupby('subspecialty')['Answer'].nunique().sort_values(ascending=False)


def create_figure_v1_grouped_bar(stats_df: pd.DataFrame) -> plt.Figure:
    """Create grouped bar chart of mean scores."""
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
    """Create heatmap of Gwet's AC1 by subspecialty x dimension."""
    setup_plotting()

    pivot = stats_df.pivot(index='subspecialty', columns='dimension', values='gwet_ac1')

    # Sort by overall mean AC1
    pivot['overall_mean'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('overall_mean', ascending=False).drop('overall_mean', axis=1)

    fig, ax = plt.subplots(figsize=(14, 10))

    sns.heatmap(
        pivot,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={'label': "Gwet's AC1"}
    )

    ax.set_title("Interrater Agreement (Gwet's AC1): Subspecialty × Dimension")
    ax.set_xlabel('Evaluation Dimension')
    ax.set_ylabel('Subspecialty')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    return fig


def create_figure_v3_agreement_comparison(agreement_df: pd.DataFrame) -> plt.Figure:
    """Create bar chart comparing Gwet's AC1 across subspecialties."""
    setup_plotting()

    fig, ax = plt.subplots(figsize=(12, 8))

    # Sort by AC1
    df_sorted = agreement_df.dropna(subset=['mean_gwet_ac1']).sort_values('mean_gwet_ac1', ascending=True)

    y_pos = np.arange(len(df_sorted))
    colors = [COLORS['quaternary'] if v < 0.4 else COLORS['tertiary'] if v < 0.6 else COLORS['success']
              for v in df_sorted['mean_gwet_ac1']]

    bars = ax.barh(y_pos, df_sorted['mean_gwet_ac1'], color=colors, alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{row['subspecialty']} (n={row['n_answers']})"
                        for _, row in df_sorted.iterrows()])
    ax.set_xlabel("Mean Gwet's AC1")
    ax.set_title("Interrater Agreement by Subspecialty")

    # Add interpretation lines
    ax.axvline(x=0.4, color=COLORS['tertiary'], linestyle='--', alpha=0.7, label='Fair/Moderate')
    ax.axvline(x=0.6, color=COLORS['success'], linestyle='--', alpha=0.7, label='Moderate/Substantial')
    ax.legend(loc='lower right')

    ax.set_xlim(0, 1)

    plt.tight_layout()
    return fig


def create_figure_v4_combined_metrics(stats_df: pd.DataFrame, agreement_df: pd.DataFrame) -> plt.Figure:
    """Create combined figure showing both performance and agreement."""
    setup_plotting()

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: Mean score heatmap
    pivot_score = stats_df.pivot(index='subspecialty', columns='dimension', values='mean')
    pivot_score['overall_mean'] = pivot_score.mean(axis=1)
    pivot_score = pivot_score.sort_values('overall_mean', ascending=False).drop('overall_mean', axis=1)

    sns.heatmap(
        pivot_score,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        vmin=1,
        vmax=5,
        ax=axes[0],
        cbar_kws={'label': 'Mean Score'}
    )
    axes[0].set_title('Performance (Mean Score)')
    axes[0].set_xlabel('Dimension')
    axes[0].set_ylabel('Subspecialty')
    plt.sca(axes[0])
    plt.xticks(rotation=45, ha='right')

    # Right: Agreement summary
    df_sorted = agreement_df.dropna(subset=['mean_gwet_ac1']).sort_values('mean_gwet_ac1', ascending=False)

    y_pos = np.arange(len(df_sorted))
    colors = [COLORS['quaternary'] if v < 0.4 else COLORS['tertiary'] if v < 0.6 else COLORS['success']
              for v in df_sorted['mean_gwet_ac1']]

    bars = axes[1].barh(y_pos, df_sorted['mean_gwet_ac1'], color=colors, alpha=0.8)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([f"{row['subspecialty']}" for _, row in df_sorted.iterrows()])
    axes[1].set_xlabel("Gwet's AC1")
    axes[1].set_title("Interrater Agreement")
    axes[1].axvline(x=0.4, color='gray', linestyle='--', alpha=0.5)
    axes[1].axvline(x=0.6, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlim(0, 1)

    # Add value labels
    for bar, val in zip(bars, df_sorted['mean_gwet_ac1']):
        axes[1].text(val + 0.02, bar.get_y() + bar.get_height()/2,
                     f'{val:.2f}', ha='left', va='center', fontsize=9)

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

    # Calculate statistics with Gwet's AC1
    print("\n4. Calculating subspecialty statistics with Gwet's AC1...")
    stats_df = calculate_subspecialty_stats(all_answers_df)

    # Calculate overall agreement by subspecialty
    print("\n5. Calculating overall agreement by subspecialty...")
    agreement_df = calculate_overall_agreement_by_subspecialty(all_answers_df)

    print("\n   Agreement by subspecialty:")
    for _, row in agreement_df.iterrows():
        ac1 = row['mean_gwet_ac1']
        if not np.isnan(ac1):
            print(f"   - {row['subspecialty']}: AC1={ac1:.3f} ({row['interpretation']})")

    # Save tables
    print("\n6. Saving tables...")
    stats_df.to_csv(TABLES_DIR / '06_subspecialty_analysis.csv', index=False)
    agreement_df.to_csv(TABLES_DIR / '06_subspecialty_agreement.csv', index=False)

    classification_df = all_answers_df[['Answer', 'Question', 'subspecialty']].drop_duplicates()
    classification_df.to_csv(TABLES_DIR / '06_subspecialty_classification.csv', index=False)
    print(f"   Saved to: {TABLES_DIR}")

    # Create figures
    print("\n7. Creating figures...")

    fig1 = create_figure_v1_grouped_bar(stats_df)
    save_figure_variants(fig1, '06_subspecialty_analysis', FIGURES_DIR, 1)
    print("   - Saved: Grouped bar chart (v1)")

    fig2 = create_figure_v2_heatmap(stats_df)
    save_figure_variants(fig2, '06_subspecialty_analysis', FIGURES_DIR, 2)
    print("   - Saved: Agreement heatmap (v2)")

    fig3 = create_figure_v3_agreement_comparison(agreement_df)
    save_figure_variants(fig3, '06_subspecialty_analysis', FIGURES_DIR, 3)
    print("   - Saved: Agreement comparison (v3)")

    fig4 = create_figure_v4_combined_metrics(stats_df, agreement_df)
    save_figure_variants(fig4, '06_subspecialty_analysis', FIGURES_DIR, 4)
    print("   - Saved: Combined metrics (v4)")

    # Summary
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

    overall_means = stats_df.groupby('subspecialty')['mean'].mean()
    best_subspec = overall_means.idxmax()
    worst_subspec = overall_means.idxmin()

    best_agreement = agreement_df.loc[agreement_df['mean_gwet_ac1'].idxmax()]
    worst_agreement = agreement_df.loc[agreement_df['mean_gwet_ac1'].idxmin()]

    print(f"\nKey findings:")
    print(f"  - {len(subspecialty_dist)} subspecialties identified")
    print(f"  - Best performing: {best_subspec} (mean: {overall_means[best_subspec]:.2f})")
    print(f"  - Worst performing: {worst_subspec} (mean: {overall_means[worst_subspec]:.2f})")
    print(f"  - Highest agreement: {best_agreement['subspecialty']} (AC1: {best_agreement['mean_gwet_ac1']:.3f})")
    print(f"  - Lowest agreement: {worst_agreement['subspecialty']} (AC1: {worst_agreement['mean_gwet_ac1']:.3f})")


if __name__ == '__main__':
    main()
