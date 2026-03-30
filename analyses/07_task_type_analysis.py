#!/usr/bin/env python3
"""
07_task_type_analysis.py
========================
Analysis 6: Performance by task type (diagnosis, prognosis, treatment, knowledge).

This script uses NLP to classify questions into task types and compares
performance across these categories using concatenated answer-level data.
Uses irrCAC to compute Gwet's AC1 as the primary agreement metric.

Output:
- output/tables/07_task_type_analysis.csv
- output/figures/07_task_type_analysis_v[1-4].png
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
    load_data, create_concatenated_answers_df, classify_task_type,
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
    """Classify all answers by task type based on their questions."""

    df = all_answers_df.copy()

    # Classify unique questions
    unique_questions = df[['Question']].drop_duplicates()
    unique_questions['task_type'] = unique_questions['Question'].apply(classify_task_type)

    # Map back to all answers
    task_map = unique_questions.set_index('Question')['task_type'].to_dict()
    df['task_type'] = df['Question'].map(task_map)

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


def calculate_task_type_stats(all_answers_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean scores and Gwet's AC1 per dimension per task type."""

    results = []

    for task_type in all_answers_df['task_type'].unique():
        df_task = all_answers_df[all_answers_df['task_type'] == task_type]
        n_answers = df_task['Answer'].nunique()
        n_ratings = len(df_task)

        for domain in EVAL_COLS:
            dim_name = domain.replace('Eval ', '')
            values = df_task[domain].dropna()

            if len(values) > 0:
                # Compute agreement
                agreement = compute_agreement_for_dimension(df_task, domain)

                results.append({
                    'task_type': task_type,
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


MIN_N_FOR_INTERPRETATION = 20  # Minimum answers for reliable AC1 interpretation


def calculate_overall_agreement_by_task_type(all_answers_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate overall Gwet's AC1 per task type (across all dimensions)."""

    results = []

    for task_type in all_answers_df['task_type'].unique():
        df_task = all_answers_df[all_answers_df['task_type'] == task_type]
        n_answers = df_task['Answer'].nunique()

        # Compute AC1 for each dimension, collect values and CIs
        ac1_values = []
        ci_lowers = []
        ci_uppers = []
        for domain in EVAL_COLS:
            agreement = compute_agreement_for_dimension(df_task, domain)
            if not np.isnan(agreement['gwet_ac1']):
                ac1_values.append(agreement['gwet_ac1'])
                ci_lowers.append(agreement['ci_lower'])
                ci_uppers.append(agreement['ci_upper'])

        mean_ac1 = np.mean(ac1_values) if ac1_values else np.nan
        mean_ci_lower = np.mean(ci_lowers) if ci_lowers else np.nan
        mean_ci_upper = np.mean(ci_uppers) if ci_uppers else np.nan

        # Only provide interpretation if sufficient sample size
        if np.isnan(mean_ac1):
            interp = 'N/A'
        elif n_answers < MIN_N_FOR_INTERPRETATION:
            interp = 'Insufficient data'
        else:
            interp = interpret_agreement(mean_ac1)

        results.append({
            'task_type': task_type,
            'n_answers': n_answers,
            'n_ratings': len(df_task),
            'mean_gwet_ac1': mean_ac1,
            'ci_lower': mean_ci_lower,
            'ci_upper': mean_ci_upper,
            'interpretation': interp
        })

    return pd.DataFrame(results).sort_values('n_answers', ascending=False)


def get_task_type_distribution(all_answers_df: pd.DataFrame) -> pd.Series:
    """Get distribution of task types by unique answers."""
    return all_answers_df.groupby('task_type')['Answer'].nunique().sort_values(ascending=False)


def create_figure_v1_grouped_bar(stats_df: pd.DataFrame) -> plt.Figure:
    """Create grouped bar chart of mean scores."""
    setup_plotting()

    task_types = stats_df['task_type'].unique().tolist()
    dimensions = stats_df['dimension'].unique()

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(dimensions))
    width = 0.15

    task_colors = {
        'Diagnosis': COLORS['primary'],
        'Prognosis': COLORS['secondary'],
        'Treatment': COLORS['success'],
        'Knowledge': COLORS['tertiary'],
        'Other': COLORS['neutral']
    }

    for i, task_type in enumerate(task_types):
        df_task = stats_df[stats_df['task_type'] == task_type]
        means = [df_task[df_task['dimension'] == d]['mean'].values[0]
                 if len(df_task[df_task['dimension'] == d]) > 0 else 0
                 for d in dimensions]
        color = task_colors.get(task_type, PALETTE[i])
        ax.bar(x + i * width, means, width, label=task_type, color=color, alpha=0.8)

    ax.set_xticks(x + width * (len(task_types) - 1) / 2)
    ax.set_xticklabels(dimensions, rotation=45, ha='right')
    ax.set_ylabel('Mean Score (1-5)')
    ax.set_xlabel('Evaluation Dimension')
    ax.set_title('Performance by Dimension and Task Type')
    ax.legend(title='Task Type')
    ax.set_ylim(0, 5.5)
    ax.axhline(y=3, color=COLORS['neutral'], linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig


def create_figure_v2_performance_heatmap(stats_df: pd.DataFrame) -> plt.Figure:
    """Create heatmap of mean scores by task type x dimension."""
    setup_plotting()

    pivot = stats_df.pivot(index='task_type', columns='dimension', values='mean')
    pivot['overall_mean'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('overall_mean', ascending=False).drop('overall_mean', axis=1)

    fig, ax = plt.subplots(figsize=(14, 6))

    sns.heatmap(
        pivot,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        vmin=1,
        vmax=5,
        ax=ax,
        cbar_kws={'label': 'Mean Score (1-5)'}
    )

    ax.set_title('Performance by Task Type and Dimension (Mean Likert Score)')
    ax.set_xlabel('Evaluation Dimension')
    ax.set_ylabel('Task Type')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    return fig


def create_figure_v3_radar(stats_df: pd.DataFrame) -> plt.Figure:
    """Create radar chart comparing task type performance profiles."""
    setup_plotting()

    task_types = stats_df.groupby('task_type')['n_ratings'].sum().sort_values(ascending=False).index.tolist()
    dimensions = stats_df['dimension'].unique().tolist()

    task_colors = {
        'Diagnosis': COLORS['primary'],
        'Prognosis': COLORS['secondary'],
        'Treatment': COLORS['success'],
        'Knowledge': COLORS['tertiary'],
        'Other': COLORS['neutral']
    }

    n = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for task_type in task_types:
        df_t = stats_df[stats_df['task_type'] == task_type]
        values = [df_t[df_t['dimension'] == d]['mean'].values[0]
                  if len(df_t[df_t['dimension'] == d]) > 0 else np.nan
                  for d in dimensions]
        values += values[:1]
        color = task_colors.get(task_type, COLORS['neutral'])
        ax.plot(angles, values, 'o-', linewidth=2, label=task_type,
                color=color, markersize=5)
        ax.fill(angles, values, alpha=0.05, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, size=8)
    ax.set_ylim(1, 5)
    ax.set_yticks([2, 3, 4, 5])

    ref = [3] * (n + 1)
    ax.plot(angles, ref, '--', linewidth=1, color=COLORS['neutral'], alpha=0.5)

    ax.set_title('Performance Profiles by Task Type', y=1.12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.05), fontsize=9)

    plt.tight_layout()
    return fig


def create_figure_v4_overall_bar(stats_df: pd.DataFrame) -> plt.Figure:
    """Create horizontal bar chart of overall mean score per task type."""
    setup_plotting()

    task_colors = {
        'Diagnosis': COLORS['primary'],
        'Prognosis': COLORS['secondary'],
        'Treatment': COLORS['success'],
        'Knowledge': COLORS['tertiary'],
        'Other': COLORS['neutral']
    }

    overall = stats_df.groupby('task_type').agg(
        mean=('mean', 'mean'),
        std=('mean', 'std'),
        n_answers=('n_answers', 'first')
    ).sort_values('mean', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    y = np.arange(len(overall))
    colors = [task_colors.get(idx, COLORS['neutral']) for idx in overall.index]
    bars = ax.barh(y, overall['mean'], xerr=overall['std'], capsize=4,
                   color=colors, alpha=0.85)

    for i, (idx, row) in enumerate(overall.iterrows()):
        ax.text(row['mean'] + row['std'] + 0.05, i,
                f"{row['mean']:.2f} (n={int(row['n_answers'])})",
                va='center', fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(overall.index)
    ax.set_xlabel('Mean Score Across All Dimensions (1-5)')
    ax.set_title('Overall Performance by Task Type (Mean ± SD)')
    ax.set_xlim(1, 5)
    ax.axvline(x=3, color=COLORS['neutral'], linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig


def main():
    """Main function for task type analysis."""

    print("=" * 60)
    print("07_task_type_analysis.py - Performance by Task Type")
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
    print("\n3. Classifying answers by task type...")
    all_answers_df = classify_answers(all_answers_df)

    task_dist = get_task_type_distribution(all_answers_df)
    print("\n   Task type distribution (unique answers):")
    for task_type, count in task_dist.items():
        print(f"   - {task_type}: {count}")

    # Calculate statistics with Gwet's AC1
    print("\n4. Calculating task type statistics with Gwet's AC1...")
    stats_df = calculate_task_type_stats(all_answers_df)

    # Calculate overall agreement by task type
    print("\n5. Calculating overall agreement by task type...")
    agreement_df = calculate_overall_agreement_by_task_type(all_answers_df)

    print("\n   Agreement by task type:")
    for _, row in agreement_df.iterrows():
        ac1 = row['mean_gwet_ac1']
        if not np.isnan(ac1):
            print(f"   - {row['task_type']}: AC1={ac1:.3f} ({row['interpretation']})")

    # Create performance summary table
    print("\n6. Creating performance summary...")
    perf_summary = stats_df.pivot(index='task_type', columns='dimension', values='mean')
    perf_summary['Overall Mean'] = perf_summary.mean(axis=1)
    perf_summary['n_answers'] = stats_df.groupby('task_type')['n_answers'].first().values
    perf_summary = perf_summary.sort_values('n_answers', ascending=False)
    perf_summary = perf_summary.round(2)

    print("\n   Performance by task type (overall mean):")
    for idx, row in perf_summary.iterrows():
        print(f"   - {idx}: {row['Overall Mean']:.2f} (n={int(row['n_answers'])})")

    # Save tables
    print("\n7. Saving tables...")
    stats_df.to_csv(TABLES_DIR / '07_task_type_analysis.csv', index=False)
    agreement_df.to_csv(TABLES_DIR / '07_task_type_agreement.csv', index=False)
    perf_summary.to_csv(TABLES_DIR / '07_task_type_performance_summary.csv')

    classification_df = all_answers_df[['Answer', 'Question', 'task_type']].drop_duplicates()
    classification_df.to_csv(TABLES_DIR / '07_task_type_classification.csv', index=False)
    print(f"   Saved to: {TABLES_DIR}")

    # Create figures
    print("\n8. Creating figures...")

    fig1 = create_figure_v1_grouped_bar(stats_df)
    save_figure_variants(fig1, '07_task_type_analysis', FIGURES_DIR, 1)
    print("   - Saved: Grouped bar chart (v1)")

    fig2 = create_figure_v2_performance_heatmap(stats_df)
    save_figure_variants(fig2, '07_task_type_analysis', FIGURES_DIR, 2)
    print("   - Saved: Performance heatmap (v2)")

    fig3 = create_figure_v3_radar(stats_df)
    save_figure_variants(fig3, '07_task_type_analysis', FIGURES_DIR, 3)
    print("   - Saved: Radar chart (v3)")

    fig4 = create_figure_v4_overall_bar(stats_df)
    save_figure_variants(fig4, '07_task_type_analysis', FIGURES_DIR, 4)
    print("   - Saved: Overall performance bar (v4)")

    # Summary
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

    overall_means = stats_df.groupby('task_type')['mean'].mean()
    best_task = overall_means.idxmax()
    worst_task = overall_means.idxmin()

    best_agreement = agreement_df.loc[agreement_df['mean_gwet_ac1'].idxmax()]
    worst_agreement = agreement_df.loc[agreement_df['mean_gwet_ac1'].idxmin()]

    print(f"\nKey findings:")
    print(f"  - {len(task_dist)} task types identified")
    print(f"  - Best performing: {best_task} (mean: {overall_means[best_task]:.2f})")
    print(f"  - Worst performing: {worst_task} (mean: {overall_means[worst_task]:.2f})")
    print(f"  - Highest agreement: {best_agreement['task_type']} (AC1: {best_agreement['mean_gwet_ac1']:.3f})")
    print(f"  - Lowest agreement: {worst_agreement['task_type']} (AC1: {worst_agreement['mean_gwet_ac1']:.3f})")


if __name__ == '__main__':
    main()
