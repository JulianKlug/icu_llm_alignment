#!/usr/bin/env python3
"""
02_vote_agreement.py
====================
Analysis 1: Interrater agreement for preferred answer over all questions.

This script calculates Fleiss' Kappa, Krippendorff's Alpha, and Gwet's AC1
for the vote (preferred answer) across all raters and questions using irrCAC.

Votes are remapped: 1 (First) -> -1, 2 (Second) -> +1, 12 (Both) -> 0

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
from irrCAC.raw import CAC

from analyses.utils import (
    load_data, setup_plotting, save_figure_variants, COLORS, PALETTE
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


def create_vote_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create vote matrix with questions as rows and raters as columns.
    Votes are remapped: 1 -> -1, 2 -> +1, 12 -> 0
    """
    df_copy = df.copy()

    # Remap votes
    df_copy['Vote'] = df_copy['Vote'].map({1: -1, 2: +1, 12: 0})

    # Pivot: rows=questions, columns=raters, values=votes
    vote_matrix = df_copy.pivot_table(
        index='Question',
        columns='Name',
        values='Vote',
        aggfunc='first'  # In case of duplicates
    )

    return vote_matrix


def calculate_agreement_metrics(vote_matrix: pd.DataFrame) -> dict:
    """Calculate agreement metrics using irrCAC."""

    cac = CAC(vote_matrix)

    # Fleiss' Kappa
    fleiss_result = cac.fleiss()
    fleiss_est = fleiss_result['est']

    # Krippendorff's Alpha
    krippendorff_result = cac.krippendorff()
    krippendorff_est = krippendorff_result['est']

    # Gwet's AC1
    gwet_result = cac.gwet()
    gwet_est = gwet_result['est']

    return {
        'fleiss_kappa': fleiss_est['coefficient_value'],
        'fleiss_ci_lower': fleiss_est['confidence_interval'][0],
        'fleiss_ci_upper': fleiss_est['confidence_interval'][1],
        'fleiss_p_value': fleiss_est['p_value'],
        'fleiss_se': fleiss_est['se'],
        'fleiss_pa': fleiss_est['pa'],
        'fleiss_pe': fleiss_est['pe'],

        'krippendorff_alpha': krippendorff_est['coefficient_value'],
        'krippendorff_ci_lower': krippendorff_est['confidence_interval'][0],
        'krippendorff_ci_upper': krippendorff_est['confidence_interval'][1],
        'krippendorff_p_value': krippendorff_est['p_value'],
        'krippendorff_se': krippendorff_est['se'],

        'gwet_ac1': gwet_est['coefficient_value'],
        'gwet_ci_lower': gwet_est['confidence_interval'][0],
        'gwet_ci_upper': gwet_est['confidence_interval'][1],
        'gwet_p_value': gwet_est['p_value'],
        'gwet_se': gwet_est['se'],

        'n_questions': vote_matrix.shape[0],
        'n_raters': vote_matrix.shape[1],
        'categories': fleiss_result['categories'],
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
    """Create summary figure with all agreement metrics."""
    setup_plotting()

    fig = plt.figure(figsize=(14, 10))

    # Create grid
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

    # Agreement coefficients comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])

    metrics = ['Fleiss\' Kappa', 'Krippendorff\'s Alpha', 'Gwet\'s AC1']
    values = [results['fleiss_kappa'], results['krippendorff_alpha'], results['gwet_ac1']]
    errors_lower = [
        results['fleiss_kappa'] - results['fleiss_ci_lower'],
        results['krippendorff_alpha'] - results['krippendorff_ci_lower'],
        results['gwet_ac1'] - results['gwet_ci_lower']
    ]
    errors_upper = [
        results['fleiss_ci_upper'] - results['fleiss_kappa'],
        results['krippendorff_ci_upper'] - results['krippendorff_alpha'],
        results['gwet_ci_upper'] - results['gwet_ac1']
    ]

    y_pos = np.arange(len(metrics))
    colors_bar = [COLORS['primary'], COLORS['secondary'], COLORS['success']]

    bars = ax1.barh(y_pos, values, color=colors_bar, alpha=0.8, height=0.5)
    ax1.errorbar(values, y_pos, xerr=[errors_lower, errors_upper],
                 fmt='none', color='black', capsize=5)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(metrics)
    ax1.axvline(x=0, color=COLORS['neutral'], linestyle='-', alpha=0.3)
    ax1.axvline(x=0.4, color=COLORS['tertiary'], linestyle='--', alpha=0.5)
    ax1.axvline(x=0.6, color=COLORS['success'], linestyle='--', alpha=0.5)
    ax1.set_xlim(-0.1, 0.8)
    ax1.set_xlabel('Coefficient Value')
    ax1.set_title('Agreement Coefficients Comparison')

    # Add value labels
    for bar, val in zip(bars, values):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', ha='left', va='center', fontsize=10)

    # Metrics table (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')

    table_data = [
        ['Metric', 'Value', '95% CI', 'p-value', 'Interpretation'],
        ['Fleiss\' Kappa',
         f'{results["fleiss_kappa"]:.3f}',
         f'({results["fleiss_ci_lower"]:.3f}, {results["fleiss_ci_upper"]:.3f})',
         f'{results["fleiss_p_value"]:.2e}',
         interpret_agreement(results["fleiss_kappa"])],
        ['Krippendorff\'s α',
         f'{results["krippendorff_alpha"]:.3f}',
         f'({results["krippendorff_ci_lower"]:.3f}, {results["krippendorff_ci_upper"]:.3f})',
         f'{results["krippendorff_p_value"]:.2e}',
         interpret_agreement(results["krippendorff_alpha"])],
        ['Gwet\'s AC1',
         f'{results["gwet_ac1"]:.3f}',
         f'({results["gwet_ci_lower"]:.3f}, {results["gwet_ci_upper"]:.3f})',
         f'{results["gwet_p_value"]:.2e}',
         interpret_agreement(results["gwet_ac1"])],
    ]

    table = ax2.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.22, 0.12, 0.22, 0.16, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # Style header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor(COLORS['light'])

    ax2.set_title('Agreement Metrics Summary', pad=20)

    # Interpretation guide (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')

    guide_data = [
        ['Range', 'Interpretation'],
        ['< 0.00', 'Poor'],
        ['0.00 - 0.20', 'Slight'],
        ['0.21 - 0.40', 'Fair'],
        ['0.41 - 0.60', 'Moderate'],
        ['0.61 - 0.80', 'Substantial'],
        ['0.81 - 1.00', 'Almost Perfect'],
    ]

    guide_table = ax3.table(cellText=guide_data, loc='center', cellLoc='center',
                            colWidths=[0.3, 0.3])
    guide_table.auto_set_font_size(False)
    guide_table.set_fontsize(10)
    guide_table.scale(1.2, 1.5)

    for (row, col), cell in guide_table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor(COLORS['light'])

    ax3.set_title('Landis & Koch Interpretation Guide', pad=20)

    # Study info (middle right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    info_text = f"""Study Information:

    • N Questions: {results['n_questions']}
    • N Raters: {results['n_raters']}
    • Vote Categories: {results['categories']}
      (-1 = First, 0 = Both, +1 = Second)

    • Observed Agreement (Pa): {results['fleiss_pa']:.3f}
    • Expected Agreement (Pe): {results['fleiss_pe']:.3f}
    """

    ax4.text(0.1, 0.5, info_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor=COLORS['light'], alpha=0.5))
    ax4.set_title('Study Details', pad=20)

    # Rater coverage (bottom)
    ax5 = fig.add_subplot(gs[2, :])
    x = np.arange(len(rater_stats))
    bars = ax5.bar(x, rater_stats['n_ratings'].values, color=PALETTE[:len(rater_stats)], alpha=0.8)
    ax5.set_xticks(x)
    ax5.set_xticklabels(rater_stats['rater'].values, rotation=45, ha='right')
    ax5.set_ylabel('Number of Ratings')
    ax5.set_xlabel('Rater')
    ax5.set_title('Ratings per Rater')

    for bar, val in zip(bars, rater_stats['n_ratings'].values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
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

    # Create vote matrix
    print("\n2. Creating vote matrix...")
    vote_matrix = create_vote_matrix(df)
    print(f"   Matrix shape: {vote_matrix.shape[0]} questions × {vote_matrix.shape[1]} raters")

    # Calculate agreement metrics using irrCAC
    print("\n3. Calculating agreement metrics (irrCAC)...")
    results = calculate_agreement_metrics(vote_matrix)

    print(f"\n   Fleiss' Kappa: {results['fleiss_kappa']:.3f}")
    print(f"     95% CI: ({results['fleiss_ci_lower']:.3f}, {results['fleiss_ci_upper']:.3f})")
    print(f"     p-value: {results['fleiss_p_value']:.2e}")
    print(f"     Interpretation: {interpret_agreement(results['fleiss_kappa'])}")

    print(f"\n   Krippendorff's Alpha: {results['krippendorff_alpha']:.3f}")
    print(f"     95% CI: ({results['krippendorff_ci_lower']:.3f}, {results['krippendorff_ci_upper']:.3f})")
    print(f"     p-value: {results['krippendorff_p_value']:.2e}")
    print(f"     Interpretation: {interpret_agreement(results['krippendorff_alpha'])}")

    print(f"\n   Gwet's AC1: {results['gwet_ac1']:.3f}")
    print(f"     95% CI: ({results['gwet_ci_lower']:.3f}, {results['gwet_ci_upper']:.3f})")
    print(f"     p-value: {results['gwet_p_value']:.2e}")
    print(f"     Interpretation: {interpret_agreement(results['gwet_ac1'])}")

    # Pairwise agreement
    print("\n4. Calculating pairwise agreement...")
    pairwise_df = calculate_pairwise_agreement(vote_matrix)

    # Rater statistics
    print("\n5. Calculating per-rater statistics...")
    rater_stats = calculate_rater_stats(df)

    # Save tables
    print("\n6. Saving tables...")

    # Main results table
    results_df = pd.DataFrame([
        {
            'metric': "Fleiss' Kappa",
            'value': results['fleiss_kappa'],
            'ci_lower': results['fleiss_ci_lower'],
            'ci_upper': results['fleiss_ci_upper'],
            'p_value': results['fleiss_p_value'],
            'se': results['fleiss_se'],
            'interpretation': interpret_agreement(results['fleiss_kappa'])
        },
        {
            'metric': "Krippendorff's Alpha",
            'value': results['krippendorff_alpha'],
            'ci_lower': results['krippendorff_ci_lower'],
            'ci_upper': results['krippendorff_ci_upper'],
            'p_value': results['krippendorff_p_value'],
            'se': results['krippendorff_se'],
            'interpretation': interpret_agreement(results['krippendorff_alpha'])
        },
        {
            'metric': "Gwet's AC1",
            'value': results['gwet_ac1'],
            'ci_lower': results['gwet_ci_lower'],
            'ci_upper': results['gwet_ci_upper'],
            'p_value': results['gwet_p_value'],
            'se': results['gwet_se'],
            'interpretation': interpret_agreement(results['gwet_ac1'])
        }
    ])

    results_df.to_csv(TABLES_DIR / '02_vote_agreement.csv', index=False)
    pairwise_df.to_csv(TABLES_DIR / '02_vote_agreement_pairwise.csv')
    rater_stats.to_csv(TABLES_DIR / '02_vote_agreement_raters.csv', index=False)
    vote_matrix.to_csv(TABLES_DIR / '02_vote_agreement_matrix.csv')
    print(f"   Saved to: {TABLES_DIR}")

    # Create figures
    print("\n7. Creating figures...")

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
    print(f"  - Fleiss' Kappa: {results['fleiss_kappa']:.3f} ({interpret_agreement(results['fleiss_kappa'])})")
    print(f"  - Krippendorff's Alpha: {results['krippendorff_alpha']:.3f} ({interpret_agreement(results['krippendorff_alpha'])})")
    print(f"  - Gwet's AC1: {results['gwet_ac1']:.3f} ({interpret_agreement(results['gwet_ac1'])})")
    print(f"  - {results['n_questions']} questions rated by {results['n_raters']} raters")


if __name__ == '__main__':
    main()
