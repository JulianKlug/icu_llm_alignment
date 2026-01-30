#!/usr/bin/env python3
"""
07_task_type_analysis.py
========================
Analysis 6: Performance by task type (diagnosis, prognosis, treatment, knowledge).

This script uses NLP to classify questions into task types and compares
performance across these categories using concatenated answer-level data.

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

from analyses.utils import (
    load_data, create_concatenated_answers_df, classify_task_type,
    setup_plotting, save_figure_variants, COLORS, PALETTE, EVAL_COLS
)

OUTPUT_DIR = project_root / 'output'
TABLES_DIR = OUTPUT_DIR / 'tables'
FIGURES_DIR = OUTPUT_DIR / 'figures'


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


def calculate_task_type_stats(all_answers_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean scores per dimension per task type."""

    results = []

    for task_type in all_answers_df['task_type'].unique():
        df_task = all_answers_df[all_answers_df['task_type'] == task_type]
        n_answers = df_task['Answer'].nunique()
        n_ratings = len(df_task)

        for domain in EVAL_COLS:
            dim_name = domain.replace('Eval ', '')
            values = df_task[domain].dropna()

            if len(values) > 0:
                results.append({
                    'task_type': task_type,
                    'dimension': dim_name,
                    'mean': values.mean(),
                    'std': values.std(),
                    'n_ratings': len(values),
                    'n_answers': n_answers
                })

    return pd.DataFrame(results)


def get_task_type_distribution(all_answers_df: pd.DataFrame) -> pd.Series:
    """Get distribution of task types by unique answers."""
    return all_answers_df.groupby('task_type')['Answer'].nunique().sort_values(ascending=False)


def create_figure_v1_grouped_bar(stats_df: pd.DataFrame) -> plt.Figure:
    """Create grouped bar chart."""
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


def create_figure_v2_heatmap(stats_df: pd.DataFrame) -> plt.Figure:
    """Create heatmap of task type x dimension."""
    setup_plotting()

    pivot = stats_df.pivot(index='task_type', columns='dimension', values='mean')

    # Sort by overall mean
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
        cbar_kws={'label': 'Mean Score'}
    )

    ax.set_title('Performance Heatmap: Task Type × Dimension')
    ax.set_xlabel('Evaluation Dimension')
    ax.set_ylabel('Task Type')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    return fig


def create_figure_v3_radar_chart(stats_df: pd.DataFrame) -> plt.Figure:
    """Create radar chart comparing all task types."""
    setup_plotting()

    task_types = stats_df['task_type'].unique().tolist()
    dimensions = stats_df['dimension'].unique()

    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))

    n_dims = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]

    task_colors = {
        'Diagnosis': COLORS['primary'],
        'Prognosis': COLORS['secondary'],
        'Treatment': COLORS['success'],
        'Knowledge': COLORS['tertiary'],
        'Other': COLORS['neutral']
    }

    for i, task_type in enumerate(task_types):
        df_task = stats_df[stats_df['task_type'] == task_type]

        values = [df_task[df_task['dimension'] == d]['mean'].values[0]
                  if len(df_task[df_task['dimension'] == d]) > 0 else 0
                  for d in dimensions]
        values += values[:1]

        color = task_colors.get(task_type, PALETTE[i])
        ax.plot(angles, values, 'o-', linewidth=2, color=color, label=task_type)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([d[:12] for d in dimensions], size=9)
    ax.set_ylim(0, 5)
    ax.set_title('Performance Profile by Task Type', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    return fig


def create_figure_v4_violin_plots(all_answers_df: pd.DataFrame) -> plt.Figure:
    """Create faceted violin plots by task type."""
    setup_plotting()

    task_types = all_answers_df['task_type'].unique().tolist()[:5]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    task_colors = {
        'Diagnosis': COLORS['primary'],
        'Prognosis': COLORS['secondary'],
        'Treatment': COLORS['success'],
        'Knowledge': COLORS['tertiary'],
        'Other': COLORS['neutral']
    }

    for i, task_type in enumerate(task_types):
        ax = axes[i]
        df_task = all_answers_df[all_answers_df['task_type'] == task_type]

        data = [df_task[col].dropna().values for col in EVAL_COLS]
        parts = ax.violinplot(data, positions=range(len(EVAL_COLS)),
                              showmeans=True, showmedians=True)

        color = task_colors.get(task_type, PALETTE[i])
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        dim_labels = [col.replace('Eval ', '')[:8] for col in EVAL_COLS]
        ax.set_xticks(range(len(EVAL_COLS)))
        ax.set_xticklabels(dim_labels, rotation=45, ha='right', fontsize=7)
        n_answers = df_task['Answer'].nunique()
        ax.set_title(f'{task_type} (n={n_answers})', fontsize=10)
        ax.set_ylim(0.5, 5.5)
        ax.axhline(y=3, color=COLORS['neutral'], linestyle='--', alpha=0.5)

        if i % 3 == 0:
            ax.set_ylabel('Score')

    # Hide unused subplot
    if len(task_types) < 6:
        axes[5].axis('off')

    plt.suptitle('Score Distribution by Task Type', y=1.02)
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

    # Calculate statistics
    print("\n4. Calculating task type statistics...")
    stats_df = calculate_task_type_stats(all_answers_df)

    # Save tables
    print("\n5. Saving tables...")
    stats_df.to_csv(TABLES_DIR / '07_task_type_analysis.csv', index=False)

    classification_df = all_answers_df[['Answer', 'Question', 'task_type']].drop_duplicates()
    classification_df.to_csv(TABLES_DIR / '07_task_type_classification.csv', index=False)
    print(f"   Saved to: {TABLES_DIR}")

    # Create figures
    print("\n6. Creating figures...")

    fig1 = create_figure_v1_grouped_bar(stats_df)
    save_figure_variants(fig1, '07_task_type_analysis', FIGURES_DIR, 1)
    print("   - Saved: Grouped bar chart (v1)")

    fig2 = create_figure_v2_heatmap(stats_df)
    save_figure_variants(fig2, '07_task_type_analysis', FIGURES_DIR, 2)
    print("   - Saved: Heatmap (v2)")

    fig3 = create_figure_v3_radar_chart(stats_df)
    save_figure_variants(fig3, '07_task_type_analysis', FIGURES_DIR, 3)
    print("   - Saved: Radar chart (v3)")

    fig4 = create_figure_v4_violin_plots(all_answers_df)
    save_figure_variants(fig4, '07_task_type_analysis', FIGURES_DIR, 4)
    print("   - Saved: Violin plots (v4)")

    # Summary
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

    overall_means = stats_df.groupby('task_type')['mean'].mean()
    best_task = overall_means.idxmax()
    worst_task = overall_means.idxmin()

    print(f"\nKey findings:")
    print(f"  - {len(task_dist)} task types identified")
    print(f"  - Best performing: {best_task} (mean: {overall_means[best_task]:.2f})")
    print(f"  - Worst performing: {worst_task} (mean: {overall_means[worst_task]:.2f})")


if __name__ == '__main__':
    main()
