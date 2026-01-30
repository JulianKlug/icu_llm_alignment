#!/usr/bin/env python3
"""
01_overall_performance.py
=========================
Analysis 0: Overall performance in each evaluated dimension for all answers.

This script analyzes the performance of Meditron LLM across all 10 evaluation
dimensions, aggregating scores from the rated answers.

Output:
- output/tables/01_overall_performance.csv
- output/figures/01_overall_performance_v[1-4].png
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

from analyses.utils import (
    load_data, get_rated_answers, get_summary_stats, DIMENSION_NAMES,
    setup_plotting, save_figure_variants, COLORS, PALETTE
)

OUTPUT_DIR = project_root / 'output'
TABLES_DIR = OUTPUT_DIR / 'tables'
FIGURES_DIR = OUTPUT_DIR / 'figures'


def calculate_overall_stats(df_rated: pd.DataFrame) -> pd.DataFrame:
    """Calculate summary statistics for each evaluation dimension."""

    stats_list = []

    for dim in DIMENSION_NAMES:
        values = df_rated[dim].dropna()
        stats = get_summary_stats(values)
        stats['dimension'] = dim
        stats_list.append(stats)

    stats_df = pd.DataFrame(stats_list)
    stats_df = stats_df[['dimension', 'n', 'mean', 'std', 'median', 'q25', 'q75', 'min', 'max']]

    return stats_df


def create_figure_v1_bar_chart(stats_df: pd.DataFrame) -> plt.Figure:
    """Create bar chart with error bars."""
    setup_plotting()

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(stats_df))
    means = stats_df['mean'].values
    stds = stats_df['std'].values
    dims = stats_df['dimension'].values

    bars = ax.bar(x, means, yerr=stds, capsize=4, color=COLORS['primary'], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(dims, rotation=45, ha='right')
    ax.set_ylabel('Mean Score (1-5 Likert Scale)')
    ax.set_xlabel('Evaluation Dimension')
    ax.set_title('Overall Performance Across Evaluation Dimensions\n(Mean ± SD)')
    ax.set_ylim(0, 5.5)
    ax.axhline(y=3, color=COLORS['neutral'], linestyle='--', alpha=0.5, label='Neutral (3)')

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=9)

    ax.legend()
    plt.tight_layout()

    return fig


def create_figure_v2_box_plot(df_rated: pd.DataFrame) -> plt.Figure:
    """Create box plots for each dimension."""
    setup_plotting()

    fig, ax = plt.subplots(figsize=(12, 6))

    data = [df_rated[dim].dropna().values for dim in DIMENSION_NAMES]

    bp = ax.boxplot(data, labels=DIMENSION_NAMES, patch_artist=True)

    for patch, color in zip(bp['boxes'], PALETTE[:len(DIMENSION_NAMES)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Score (1-5 Likert Scale)')
    ax.set_xlabel('Evaluation Dimension')
    ax.set_title('Distribution of Scores Across Evaluation Dimensions')
    ax.axhline(y=3, color=COLORS['neutral'], linestyle='--', alpha=0.5, label='Neutral (3)')
    plt.xticks(rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()

    return fig


def create_figure_v3_violin_plot(df_rated: pd.DataFrame) -> plt.Figure:
    """Create violin plots for each dimension."""
    setup_plotting()

    fig, ax = plt.subplots(figsize=(12, 6))

    data = [df_rated[dim].dropna().values for dim in DIMENSION_NAMES]

    parts = ax.violinplot(data, positions=range(len(DIMENSION_NAMES)),
                          showmeans=True, showmedians=True)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(PALETTE[i % len(PALETTE)])
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(DIMENSION_NAMES)))
    ax.set_xticklabels(DIMENSION_NAMES, rotation=45, ha='right')
    ax.set_ylabel('Score (1-5 Likert Scale)')
    ax.set_xlabel('Evaluation Dimension')
    ax.set_title('Distribution of Scores Across Evaluation Dimensions (Violin Plot)')
    ax.axhline(y=3, color=COLORS['neutral'], linestyle='--', alpha=0.5, label='Neutral (3)')
    ax.legend()
    plt.tight_layout()

    return fig


def create_figure_v4_radar_chart(stats_df: pd.DataFrame) -> plt.Figure:
    """Create radar/spider chart."""
    setup_plotting()

    # Prepare data
    dims = stats_df['dimension'].values
    means = stats_df['mean'].values
    n_dims = len(dims)

    # Compute angles
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]

    # Complete the circle
    values = means.tolist()
    values += values[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    ax.plot(angles, values, 'o-', linewidth=2, color=COLORS['primary'], label='Mean Score')
    ax.fill(angles, values, alpha=0.25, color=COLORS['primary'])

    # Set ticks
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, size=9)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])

    # Add reference circle at 3 (neutral)
    ref_values = [3] * (n_dims + 1)
    ax.plot(angles, ref_values, '--', linewidth=1, color=COLORS['neutral'],
            alpha=0.7, label='Neutral (3)')

    ax.set_title('Overall Performance Profile\n(Mean Scores by Dimension)', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()

    return fig


def main():
    """Main function for overall performance analysis."""

    print("=" * 60)
    print("01_overall_performance.py - Overall Performance Analysis")
    print("=" * 60)

    # Create output directories
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    print("\n1. Loading data...")
    df = load_data()
    print(f"   Loaded {len(df)} ratings")

    # Get rated answers (combines first and second answer evals based on Vote)
    print("\n2. Extracting rated answers...")
    df_rated = get_rated_answers(df)
    print(f"   {len(df_rated)} answer evaluations extracted")

    # Calculate statistics
    print("\n3. Calculating summary statistics...")
    stats_df = calculate_overall_stats(df_rated)

    # Save table
    table_path = TABLES_DIR / '01_overall_performance.csv'
    stats_df.to_csv(table_path, index=False)
    print(f"   Saved table to: {table_path}")

    # Display statistics
    print("\n   Summary Statistics:")
    print(stats_df.to_string(index=False))

    # Create figures
    print("\n4. Creating figures...")

    fig1 = create_figure_v1_bar_chart(stats_df)
    save_figure_variants(fig1, '01_overall_performance', FIGURES_DIR, 1)
    print("   - Saved: Bar chart with error bars (v1)")

    fig2 = create_figure_v2_box_plot(df_rated)
    save_figure_variants(fig2, '01_overall_performance', FIGURES_DIR, 2)
    print("   - Saved: Box plots (v2)")

    fig3 = create_figure_v3_violin_plot(df_rated)
    save_figure_variants(fig3, '01_overall_performance', FIGURES_DIR, 3)
    print("   - Saved: Violin plots (v3)")

    fig4 = create_figure_v4_radar_chart(stats_df)
    save_figure_variants(fig4, '01_overall_performance', FIGURES_DIR, 4)
    print("   - Saved: Radar chart (v4)")

    # Summary
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    print(f"\nKey findings:")
    print(f"  - Highest scoring dimension: {stats_df.loc[stats_df['mean'].idxmax(), 'dimension']} ({stats_df['mean'].max():.2f})")
    print(f"  - Lowest scoring dimension: {stats_df.loc[stats_df['mean'].idxmin(), 'dimension']} ({stats_df['mean'].min():.2f})")
    print(f"  - Overall mean across dimensions: {stats_df['mean'].mean():.2f}")


if __name__ == '__main__':
    main()
