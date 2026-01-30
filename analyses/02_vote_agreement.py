#!/usr/bin/env python3
"""
02_vote_agreement.py
====================
Analysis 1: Interrater agreement for preferred answer over all questions.

This script calculates Fleiss' Kappa and other agreement metrics for the
vote (preferred answer) across all raters and questions.

Output:
- output/tables/02_vote_agreement.csv
- output/figures/02_vote_agreement_v[1-4].png
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
    load_data, reshape_for_agreement, fleiss_kappa, percent_agreement,
    bootstrap_ci, setup_plotting, save_figure_variants, COLORS, PALETTE
)
from analyses.utils.stats import interpret_kappa

OUTPUT_DIR = project_root / 'output'
TABLES_DIR = OUTPUT_DIR / 'tables'
FIGURES_DIR = OUTPUT_DIR / 'figures'


def calculate_vote_agreement(df: pd.DataFrame) -> dict:
    """Calculate interrater agreement metrics for vote."""

    # Reshape data: questions x raters
    vote_matrix = reshape_for_agreement(df, 'Vote')

    # Calculate Fleiss' Kappa
    kappa = fleiss_kappa(vote_matrix)

    # Calculate percent agreement
    pct_agree = percent_agreement(vote_matrix)

    # Bootstrap CI for Kappa
    def kappa_func(data):
        return fleiss_kappa(data)

    kappa_point, kappa_ci_lower, kappa_ci_upper = bootstrap_ci(
        vote_matrix, kappa_func, n_bootstrap=1000
    )

    return {
        'fleiss_kappa': kappa,
        'kappa_ci_lower': kappa_ci_lower,
        'kappa_ci_upper': kappa_ci_upper,
        'percent_agreement': pct_agree,
        'interpretation': interpret_kappa(kappa),
        'n_questions': vote_matrix.shape[0],
        'n_raters': vote_matrix.shape[1],
        'vote_matrix': vote_matrix
    }


def calculate_pairwise_agreement(vote_matrix: pd.DataFrame) -> pd.DataFrame:
    """Calculate pairwise agreement between all rater pairs."""

    raters = vote_matrix.columns.tolist()
    n_raters = len(raters)

    agreement_matrix = np.zeros((n_raters, n_raters))

    for i, rater1 in enumerate(raters):
        for j, rater2 in enumerate(raters):
            if i == j:
                agreement_matrix[i, j] = 1.0
            else:
                # Get common questions
                r1 = vote_matrix[rater1]
                r2 = vote_matrix[rater2]
                mask = r1.notna() & r2.notna()

                if mask.sum() > 0:
                    agreement = (r1[mask] == r2[mask]).mean()
                    agreement_matrix[i, j] = agreement
                else:
                    agreement_matrix[i, j] = np.nan

    return pd.DataFrame(agreement_matrix, index=raters, columns=raters)


def calculate_rater_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate per-rater statistics."""

    rater_stats = []

    for rater, group in df.groupby('Name'):
        vote_dist = group['Vote'].value_counts(normalize=True)
        stats = {
            'rater': rater,
            'n_ratings': len(group),
            'n_questions': group['question_id'].nunique(),
            'pct_vote_1': vote_dist.get(1, 0) * 100,
            'pct_vote_2': vote_dist.get(2, 0) * 100,
            'pct_vote_12': vote_dist.get(12, 0) * 100,
        }
        rater_stats.append(stats)

    return pd.DataFrame(rater_stats).sort_values('n_ratings', ascending=False)


def create_figure_v1_heatmap(pairwise_df: pd.DataFrame) -> plt.Figure:
    """Create heatmap of pairwise agreement."""
    setup_plotting()

    fig, ax = plt.subplots(figsize=(10, 8))

    mask = np.eye(len(pairwise_df), dtype=bool)
    sns.heatmap(
        pairwise_df,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        ax=ax,
        mask=mask,
        cbar_kws={'label': 'Agreement Rate'}
    )

    ax.set_title('Pairwise Agreement Between Raters\n(Vote Preference)')
    plt.tight_layout()

    return fig


def create_figure_v2_rater_votes(rater_stats: pd.DataFrame) -> plt.Figure:
    """Create stacked bar chart of vote distribution per rater."""
    setup_plotting()

    fig, ax = plt.subplots(figsize=(12, 6))

    raters = rater_stats['rater'].values
    x = np.arange(len(raters))
    width = 0.6

    # Stacked bars
    bottom = np.zeros(len(raters))

    colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary']]
    labels = ['Vote 1 (First)', 'Vote 2 (Second)', 'Vote 12 (Both)']

    for i, (col, color, label) in enumerate(zip(['pct_vote_1', 'pct_vote_2', 'pct_vote_12'], colors, labels)):
        values = rater_stats[col].values
        ax.bar(x, values, width, bottom=bottom, label=label, color=color, alpha=0.8)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(raters, rotation=45, ha='right')
    ax.set_ylabel('Percentage of Votes')
    ax.set_xlabel('Rater')
    ax.set_title('Vote Distribution by Rater')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 105)

    plt.tight_layout()

    return fig


def create_figure_v3_vote_distribution(df: pd.DataFrame) -> plt.Figure:
    """Create overall vote distribution pie chart and bar chart."""
    setup_plotting()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Vote counts
    vote_counts = df['Vote'].value_counts().sort_index()
    labels = ['First Answer', 'Second Answer', 'Both Equal']
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary']]

    # Pie chart
    axes[0].pie(vote_counts, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, explode=[0.02, 0.02, 0.02])
    axes[0].set_title('Overall Vote Distribution')

    # Bar chart
    x = np.arange(len(vote_counts))
    bars = axes[1].bar(x, vote_counts.values, color=colors, alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel('Number of Votes')
    axes[1].set_title('Vote Counts')

    # Add value labels
    for bar, val in zip(bars, vote_counts.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     str(val), ha='center', va='bottom', fontsize=11)

    plt.tight_layout()

    return fig


def create_figure_v4_agreement_summary(results: dict, rater_stats: pd.DataFrame) -> plt.Figure:
    """Create summary figure with Kappa and key metrics."""
    setup_plotting()

    fig = plt.figure(figsize=(14, 8))

    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Kappa gauge (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    kappa = results['fleiss_kappa']
    ci_lower = results['kappa_ci_lower']
    ci_upper = results['kappa_ci_upper']

    # Kappa bar
    ax1.barh(['Fleiss\' Kappa'], [kappa], color=COLORS['primary'], alpha=0.8, height=0.4)
    ax1.errorbar([kappa], ['Fleiss\' Kappa'],
                 xerr=[[kappa - ci_lower], [ci_upper - kappa]],
                 fmt='none', color='black', capsize=5)
    ax1.axvline(x=0, color=COLORS['neutral'], linestyle='-', alpha=0.3)
    ax1.axvline(x=0.4, color=COLORS['tertiary'], linestyle='--', alpha=0.5, label='Fair/Moderate')
    ax1.axvline(x=0.6, color=COLORS['success'], linestyle='--', alpha=0.5, label='Substantial')
    ax1.set_xlim(-0.1, 1)
    ax1.set_xlabel('Kappa Value')
    ax1.set_title(f'Fleiss\' Kappa: {kappa:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})\n{results["interpretation"]} Agreement')
    ax1.legend(loc='lower right', fontsize=8)

    # Metrics table (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')

    table_data = [
        ['Metric', 'Value'],
        ['Fleiss\' Kappa', f'{kappa:.3f}'],
        ['95% CI', f'{ci_lower:.3f} - {ci_upper:.3f}'],
        ['Interpretation', results['interpretation']],
        ['Percent Agreement', f'{results["percent_agreement"]*100:.1f}%'],
        ['N Questions', str(results['n_questions'])],
        ['N Raters', str(results['n_raters'])],
    ]

    table = ax2.table(cellText=table_data, loc='center', cellLoc='left',
                      colWidths=[0.4, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor(COLORS['light'])

    ax2.set_title('Agreement Summary', pad=20)

    # Rater coverage (bottom)
    ax3 = fig.add_subplot(gs[1, :])
    x = np.arange(len(rater_stats))
    bars = ax3.bar(x, rater_stats['n_ratings'].values, color=PALETTE[:len(rater_stats)], alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(rater_stats['rater'].values, rotation=45, ha='right')
    ax3.set_ylabel('Number of Ratings')
    ax3.set_xlabel('Rater')
    ax3.set_title('Ratings per Rater')

    for bar, val in zip(bars, rater_stats['n_ratings'].values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(val), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    return fig


def main():
    """Main function for vote agreement analysis."""

    print("=" * 60)
    print("02_vote_agreement.py - Interrater Agreement for Votes")
    print("=" * 60)

    # Create output directories
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n1. Loading data...")
    df = load_data()
    print(f"   Loaded {len(df)} ratings from {df['Name'].nunique()} raters")

    # Calculate agreement
    print("\n2. Calculating Fleiss' Kappa...")
    results = calculate_vote_agreement(df)
    print(f"   Fleiss' Kappa: {results['fleiss_kappa']:.3f}")
    print(f"   95% CI: [{results['kappa_ci_lower']:.3f}, {results['kappa_ci_upper']:.3f}]")
    print(f"   Interpretation: {results['interpretation']}")
    print(f"   Percent Agreement: {results['percent_agreement']*100:.1f}%")

    # Pairwise agreement
    print("\n3. Calculating pairwise agreement...")
    pairwise_df = calculate_pairwise_agreement(results['vote_matrix'])

    # Rater statistics
    print("\n4. Calculating per-rater statistics...")
    rater_stats = calculate_rater_stats(df)

    # Save tables
    print("\n5. Saving tables...")

    # Main results table
    results_df = pd.DataFrame([{
        'metric': 'Fleiss\' Kappa',
        'value': results['fleiss_kappa'],
        'ci_lower': results['kappa_ci_lower'],
        'ci_upper': results['kappa_ci_upper'],
        'interpretation': results['interpretation']
    }, {
        'metric': 'Percent Agreement',
        'value': results['percent_agreement'],
        'ci_lower': np.nan,
        'ci_upper': np.nan,
        'interpretation': ''
    }])

    results_df.to_csv(TABLES_DIR / '02_vote_agreement.csv', index=False)
    pairwise_df.to_csv(TABLES_DIR / '02_vote_agreement_pairwise.csv')
    rater_stats.to_csv(TABLES_DIR / '02_vote_agreement_raters.csv', index=False)
    print(f"   Saved to: {TABLES_DIR}")

    # Create figures
    print("\n6. Creating figures...")

    fig1 = create_figure_v1_heatmap(pairwise_df)
    save_figure_variants(fig1, '02_vote_agreement', FIGURES_DIR, 1)
    print("   - Saved: Pairwise agreement heatmap (v1)")

    fig2 = create_figure_v2_rater_votes(rater_stats)
    save_figure_variants(fig2, '02_vote_agreement', FIGURES_DIR, 2)
    print("   - Saved: Vote distribution by rater (v2)")

    fig3 = create_figure_v3_vote_distribution(df)
    save_figure_variants(fig3, '02_vote_agreement', FIGURES_DIR, 3)
    print("   - Saved: Overall vote distribution (v3)")

    fig4 = create_figure_v4_agreement_summary(results, rater_stats)
    save_figure_variants(fig4, '02_vote_agreement', FIGURES_DIR, 4)
    print("   - Saved: Agreement summary (v4)")

    # Summary
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    print(f"\nKey findings:")
    print(f"  - Fleiss' Kappa: {results['fleiss_kappa']:.3f} ({results['interpretation']})")
    print(f"  - {results['n_questions']} questions rated by {results['n_raters']} raters")


if __name__ == '__main__':
    main()
