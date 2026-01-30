#!/usr/bin/env python3
"""
05_correlation_analysis.py
==========================
Analysis 4: Correlation between Alignment with Guidelines and interrater agreement.

This script examines whether answers where the LLM aligns well with guidelines
also have higher interrater agreement (lower std).

Output:
- output/tables/05_correlation_analysis.csv
- output/figures/05_correlation_analysis_v[1-4].png
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
from scipy import stats

from analyses.utils import (
    load_data, create_concatenated_answers_df,
    setup_plotting, save_figure_variants, COLORS, PALETTE, EVAL_COLS
)

OUTPUT_DIR = project_root / 'output'
TABLES_DIR = OUTPUT_DIR / 'tables'
FIGURES_DIR = OUTPUT_DIR / 'figures'


def compute_answer_metrics(all_answers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean alignment and agreement (std) per answer.
    """
    results = []

    for answer in all_answers_df['Answer'].unique():
        answer_df = all_answers_df[all_answers_df['Answer'] == answer]
        question = answer_df['Question'].iloc[0]
        n_raters = len(answer_df)

        row = {
            'Answer': answer,
            'Question': question,
            'n_raters': n_raters
        }

        # Compute mean and std for each dimension
        stds = []
        for domain in EVAL_COLS:
            data = answer_df[domain].dropna()
            dim_name = domain.replace('Eval ', '')

            if len(data) >= 2:
                row[f'{dim_name}_mean'] = data.mean()
                row[f'{dim_name}_std'] = data.std()
                stds.append(data.std())
            elif len(data) == 1:
                row[f'{dim_name}_mean'] = data.mean()
                row[f'{dim_name}_std'] = np.nan
            else:
                row[f'{dim_name}_mean'] = np.nan
                row[f'{dim_name}_std'] = np.nan

        # Overall agreement score (mean std)
        row['mean_std'] = np.mean(stds) if stds else np.nan
        results.append(row)

    return pd.DataFrame(results)


def calculate_correlations(metrics_df: pd.DataFrame) -> dict:
    """Calculate correlations between alignment and agreement (std)."""

    # Get alignment mean and overall std
    alignment_col = 'Alignment with Guidelines_mean'

    clean = metrics_df[[alignment_col, 'mean_std']].dropna()

    if len(clean) < 3:
        return {'error': 'Not enough data'}

    alignment = clean[alignment_col]
    agreement_std = clean['mean_std']

    # Pearson (alignment vs std - expect negative: higher alignment = lower std = better agreement)
    pearson_r, pearson_p = stats.pearsonr(alignment, agreement_std)

    # Spearman
    spearman_r, spearman_p = stats.spearmanr(alignment, agreement_std)

    # Confidence interval for Pearson
    n = len(clean)
    z = np.arctanh(pearson_r)
    se = 1 / np.sqrt(n - 3)
    ci_lower = np.tanh(z - 1.96 * se)
    ci_upper = np.tanh(z + 1.96 * se)

    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'pearson_ci_lower': ci_lower,
        'pearson_ci_upper': ci_upper,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'n': n
    }


def create_figure_v1_scatter(metrics_df: pd.DataFrame, corr: dict) -> plt.Figure:
    """Create scatter plot of alignment vs agreement (std)."""
    setup_plotting()

    fig, ax = plt.subplots(figsize=(10, 8))

    alignment_col = 'Alignment with Guidelines_mean'
    clean = metrics_df[[alignment_col, 'mean_std']].dropna()

    ax.scatter(clean[alignment_col], clean['mean_std'],
               alpha=0.6, s=60, color=COLORS['primary'])

    # Regression line
    z = np.polyfit(clean[alignment_col], clean['mean_std'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(clean[alignment_col].min(), clean[alignment_col].max(), 100)
    ax.plot(x_line, p(x_line), '--', color=COLORS['secondary'], linewidth=2)

    ax.set_xlabel('Mean Alignment with Guidelines Score')
    ax.set_ylabel('Mean Std of Ratings (Lower = Better Agreement)')
    ax.set_title(f'Alignment vs Agreement\n'
                 f'Pearson r = {corr["pearson_r"]:.3f} (p = {corr["pearson_p"]:.3f})')

    plt.tight_layout()
    return fig


def create_figure_v2_hexbin(metrics_df: pd.DataFrame, corr: dict) -> plt.Figure:
    """Create hexbin density plot."""
    setup_plotting()

    fig, ax = plt.subplots(figsize=(10, 8))

    alignment_col = 'Alignment with Guidelines_mean'
    clean = metrics_df[[alignment_col, 'mean_std']].dropna()

    hb = ax.hexbin(clean[alignment_col], clean['mean_std'],
                   gridsize=15, cmap='YlOrRd', mincnt=1)
    plt.colorbar(hb, ax=ax, label='Count')

    ax.set_xlabel('Mean Alignment with Guidelines')
    ax.set_ylabel('Mean Std (Agreement)')
    ax.set_title(f'Density: Alignment vs Agreement\nPearson r = {corr["pearson_r"]:.3f}')

    plt.tight_layout()
    return fig


def create_figure_v3_by_dimension(metrics_df: pd.DataFrame) -> plt.Figure:
    """Create correlation matrix of all dimensions' mean vs std."""
    setup_plotting()

    fig, axes = plt.subplots(2, 5, figsize=(16, 8))
    axes = axes.flatten()

    for i, domain in enumerate(EVAL_COLS):
        ax = axes[i]
        dim_name = domain.replace('Eval ', '')
        mean_col = f'{dim_name}_mean'
        std_col = f'{dim_name}_std'

        if mean_col in metrics_df.columns and std_col in metrics_df.columns:
            clean = metrics_df[[mean_col, std_col]].dropna()

            if len(clean) > 2:
                ax.scatter(clean[mean_col], clean[std_col], alpha=0.5, s=20,
                          color=PALETTE[i % len(PALETTE)])

                # Correlation
                r, p = stats.pearsonr(clean[mean_col], clean[std_col])
                ax.set_title(f'{dim_name[:15]}\nr={r:.2f}', fontsize=9)
            else:
                ax.set_title(f'{dim_name[:15]}\nNo data', fontsize=9)

        ax.set_xlabel('Mean', fontsize=8)
        ax.set_ylabel('Std', fontsize=8)

    plt.suptitle('Mean Score vs Variability per Dimension', y=1.02)
    plt.tight_layout()
    return fig


def create_figure_v4_binned_analysis(metrics_df: pd.DataFrame) -> plt.Figure:
    """Create boxplot of agreement by alignment bins."""
    setup_plotting()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    alignment_col = 'Alignment with Guidelines_mean'
    clean = metrics_df[[alignment_col, 'mean_std']].dropna().copy()

    # Bin alignment scores
    clean['alignment_bin'] = pd.cut(clean[alignment_col],
                                     bins=[0, 2, 3, 4, 5],
                                     labels=['Low (1-2)', 'Medium (2-3)', 'Good (3-4)', 'High (4-5)'])

    # Left: scatter colored by bin
    colors_map = {'Low (1-2)': COLORS['quaternary'], 'Medium (2-3)': COLORS['tertiary'],
                  'Good (3-4)': COLORS['primary'], 'High (4-5)': COLORS['success']}

    for bin_label in ['Low (1-2)', 'Medium (2-3)', 'Good (3-4)', 'High (4-5)']:
        subset = clean[clean['alignment_bin'] == bin_label]
        if len(subset) > 0:
            axes[0].scatter(subset[alignment_col], subset['mean_std'],
                           alpha=0.6, s=60, label=bin_label, color=colors_map[bin_label])

    axes[0].set_xlabel('Mean Alignment Score')
    axes[0].set_ylabel('Mean Std (Agreement)')
    axes[0].set_title('Answers by Alignment Level')
    axes[0].legend()

    # Right: boxplot of std by alignment bin
    bins_present = [b for b in ['Low (1-2)', 'Medium (2-3)', 'Good (3-4)', 'High (4-5)']
                    if len(clean[clean['alignment_bin'] == b]) > 0]
    data = [clean[clean['alignment_bin'] == b]['mean_std'].values for b in bins_present]

    bp = axes[1].boxplot(data, patch_artist=True)
    for patch, bin_label in zip(bp['boxes'], bins_present):
        patch.set_facecolor(colors_map[bin_label])
        patch.set_alpha(0.7)

    axes[1].set_xticklabels(bins_present, rotation=45, ha='right')
    axes[1].set_ylabel('Mean Std (Lower = Better Agreement)')
    axes[1].set_title('Agreement Distribution by Alignment Level')

    plt.tight_layout()
    return fig


def main():
    """Main function for correlation analysis."""

    print("=" * 60)
    print("05_correlation_analysis.py - Alignment vs Agreement Correlation")
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

    # Compute per-answer metrics
    print("\n3. Computing per-answer metrics...")
    metrics_df = compute_answer_metrics(all_answers_df)
    print(f"   {len(metrics_df)} answers analyzed")

    # Calculate correlations
    print("\n4. Calculating correlations...")
    corr = calculate_correlations(metrics_df)

    if 'error' not in corr:
        print(f"   Pearson r: {corr['pearson_r']:.3f} (p = {corr['pearson_p']:.4f})")
        print(f"   95% CI: [{corr['pearson_ci_lower']:.3f}, {corr['pearson_ci_upper']:.3f}]")
        print(f"   Spearman ρ: {corr['spearman_r']:.3f} (p = {corr['spearman_p']:.4f})")
    else:
        print(f"   Error: {corr['error']}")

    # Save tables
    print("\n5. Saving tables...")

    if 'error' not in corr:
        corr_df = pd.DataFrame([{
            'metric': 'Pearson correlation (alignment vs std)',
            'value': corr['pearson_r'],
            'p_value': corr['pearson_p'],
            'ci_lower': corr['pearson_ci_lower'],
            'ci_upper': corr['pearson_ci_upper'],
            'interpretation': 'Negative = higher alignment → lower std → better agreement'
        }, {
            'metric': 'Spearman correlation',
            'value': corr['spearman_r'],
            'p_value': corr['spearman_p'],
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'interpretation': ''
        }])
        corr_df.to_csv(TABLES_DIR / '05_correlation_analysis.csv', index=False)

    metrics_df.to_csv(TABLES_DIR / '05_correlation_metrics.csv', index=False)
    print(f"   Saved to: {TABLES_DIR}")

    # Create figures
    print("\n6. Creating figures...")

    if 'error' not in corr:
        fig1 = create_figure_v1_scatter(metrics_df, corr)
        save_figure_variants(fig1, '05_correlation_analysis', FIGURES_DIR, 1)
        print("   - Saved: Scatter plot (v1)")

        fig2 = create_figure_v2_hexbin(metrics_df, corr)
        save_figure_variants(fig2, '05_correlation_analysis', FIGURES_DIR, 2)
        print("   - Saved: Hexbin density (v2)")

    fig3 = create_figure_v3_by_dimension(metrics_df)
    save_figure_variants(fig3, '05_correlation_analysis', FIGURES_DIR, 3)
    print("   - Saved: By dimension (v3)")

    fig4 = create_figure_v4_binned_analysis(metrics_df)
    save_figure_variants(fig4, '05_correlation_analysis', FIGURES_DIR, 4)
    print("   - Saved: Binned analysis (v4)")

    # Summary
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

    if 'error' not in corr:
        sig = "significant" if corr['pearson_p'] < 0.05 else "not significant"
        direction = "negative (higher alignment → better agreement)" if corr['pearson_r'] < 0 else "positive"
        print(f"\nKey findings:")
        print(f"  - Correlation is {sig} (p = {corr['pearson_p']:.4f})")
        print(f"  - Direction: {direction}")
        print(f"  - Pearson r = {corr['pearson_r']:.3f}")


if __name__ == '__main__':
    main()
