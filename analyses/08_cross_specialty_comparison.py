#!/usr/bin/env python3
"""
08_cross_specialty_comparison.py
================================
Supplementary Analysis: Cross-specialty comparison of Meditron-3 (70B) base model
performance between ICU and Nuclear Medicine evaluations.

The Nuclear Medicine evaluation data comes from:
  Bongartz G, et al. Single-center evaluation of Meditron in nuclear medicine
  clinical real-world scenarios. J Nucl Med. 2025;66(Suppl 1):251801.

Since only percentage-positive ratings are available from the conference abstract,
we compute the same metric (% positive = score >= 4, % neutral = score 3,
% negative = score <= 2) from our ICU data for a like-for-like comparison.

Output:
- output/tables/08_cross_specialty_comparison.csv
- output/figures/08_cross_specialty_comparison_v[1-4].png
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

from analyses.utils import (
    load_data, get_rated_answers, DIMENSION_NAMES,
    setup_plotting, save_figure_variants, COLORS, PALETTE
)

OUTPUT_DIR = project_root / 'output'
TABLES_DIR = OUTPUT_DIR / 'tables'
FIGURES_DIR = OUTPUT_DIR / 'figures'

# Nuclear Medicine data from J Nucl Med 2025;66(Suppl 1):251801
# 76 reviews, 36 adversarial questions, 3 raters
# Reported as % positive (score >= 4), % neutral (score = 3)
# % negative inferred as 100 - positive - neutral where available
NUCLEAR_MED_DATA = {
    'Communication & Clarity':        {'positive': 55, 'neutral': None},
    'Fairness':                       {'positive': 53, 'neutral': None},
    'Harmlessness':                   {'positive': None, 'neutral': 49},
    'Alignment with Guidelines':      {'positive': None, 'neutral': 50},
    'Logical Reasoning':              {'positive': None, 'neutral': 50},
    'Relevance & Completeness':       {'positive': 30, 'neutral': None},
    'Rater Confidence':               {'positive': 32, 'neutral': None},
    # Dimensions not individually reported in the abstract:
    'Question Comprehension':         {'positive': None, 'neutral': None},
    'Contextual Awareness':           {'positive': None, 'neutral': None},
    'Model Confidence':               {'positive': None, 'neutral': None},
}

NUCLEAR_MED_META = {
    'n_evaluations': 76,
    'n_questions': 36,
    'n_raters': 3,
    'specialty': 'Nuclear Medicine',
    'reference': 'Bongartz et al. J Nucl Med. 2025;66(Suppl 1):251801',
}


def compute_icu_percentages(df_rated: pd.DataFrame) -> pd.DataFrame:
    """
    Compute % positive (>= 4), % neutral (== 3), and % negative (<= 2)
    for each dimension from ICU evaluation data.
    """
    results = []
    for dim in DIMENSION_NAMES:
        values = df_rated[dim].dropna()
        n = len(values)
        if n == 0:
            continue

        n_positive = (values >= 4).sum()
        n_neutral = (values == 3).sum()
        n_negative = (values <= 2).sum()

        results.append({
            'dimension': dim,
            'n': n,
            'positive_pct': round(100 * n_positive / n, 1),
            'neutral_pct': round(100 * n_neutral / n, 1),
            'negative_pct': round(100 * n_negative / n, 1),
            'mean': values.mean(),
            'std': values.std(),
        })

    return pd.DataFrame(results)


def build_comparison_table(icu_pct: pd.DataFrame) -> pd.DataFrame:
    """Build side-by-side comparison table."""
    rows = []
    for _, row in icu_pct.iterrows():
        dim = row['dimension']
        nm = NUCLEAR_MED_DATA.get(dim, {})

        rows.append({
            'Dimension': dim,
            'ICU_n': int(row['n']),
            'ICU_positive_pct': row['positive_pct'],
            'ICU_neutral_pct': row['neutral_pct'],
            'ICU_negative_pct': row['negative_pct'],
            'ICU_mean': round(row['mean'], 2),
            'ICU_std': round(row['std'], 2),
            'NucMed_positive_pct': nm.get('positive'),
            'NucMed_neutral_pct': nm.get('neutral'),
            'NucMed_n': NUCLEAR_MED_META['n_evaluations'],
            'Difference_positive_pct': (
                round(row['positive_pct'] - nm['positive'], 1)
                if nm.get('positive') is not None else None
            ),
        })

    df = pd.DataFrame(rows)

    # Sort by ICU positive % descending for readability
    df = df.sort_values('ICU_positive_pct', ascending=False).reset_index(drop=True)

    return df


def create_figure_v1_grouped_bar(comparison_df: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart: ICU vs Nuclear Medicine % positive ratings."""
    setup_plotting()

    # Only include dimensions where nuclear medicine data is available
    plot_df = comparison_df.dropna(subset=['NucMed_positive_pct']).copy()
    plot_df = plot_df.sort_values('ICU_positive_pct', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    y = np.arange(len(plot_df))
    height = 0.35

    bars_icu = ax.barh(y + height/2, plot_df['ICU_positive_pct'], height,
                       label=f'ICU (n={plot_df["ICU_n"].iloc[0]})',
                       color=COLORS['primary'], alpha=0.85)
    bars_nm = ax.barh(y - height/2, plot_df['NucMed_positive_pct'], height,
                      label=f'Nuclear Medicine (n={NUCLEAR_MED_META["n_evaluations"]})',
                      color=COLORS['tertiary'], alpha=0.85)

    # Add value labels
    for bar in bars_icu:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{width:.0f}%', va='center', fontsize=9)
    for bar in bars_nm:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{width:.0f}%', va='center', fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df['Dimension'])
    ax.set_xlabel('% Positive Ratings (Score ≥ 4)')
    ax.set_title('Meditron-3 (70B) Base Model: ICU vs Nuclear Medicine\n'
                 '% Positive Ratings by Evaluation Dimension')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 100)
    ax.axvline(x=50, color=COLORS['neutral'], linestyle='--', alpha=0.4)

    plt.tight_layout()
    return fig


def create_figure_v2_stacked_bar(icu_pct: pd.DataFrame) -> plt.Figure:
    """Stacked bar chart showing positive/neutral/negative for ICU data,
    with nuclear medicine positive % overlaid as markers."""
    setup_plotting()

    plot_df = icu_pct.sort_values('positive_pct', ascending=True).copy()

    fig, ax = plt.subplots(figsize=(10, 7))

    y = np.arange(len(plot_df))

    # Stacked bars for ICU
    ax.barh(y, plot_df['positive_pct'], label='Positive (≥4)',
            color='#2ecc71', alpha=0.8)
    ax.barh(y, plot_df['neutral_pct'], left=plot_df['positive_pct'],
            label='Neutral (3)', color='#f39c12', alpha=0.8)
    ax.barh(y, plot_df['negative_pct'],
            left=plot_df['positive_pct'] + plot_df['neutral_pct'],
            label='Negative (≤2)', color='#e74c3c', alpha=0.8)

    # Overlay nuclear medicine positive % as markers
    for i, dim in enumerate(plot_df['dimension']):
        nm = NUCLEAR_MED_DATA.get(dim, {})
        if nm.get('positive') is not None:
            ax.scatter(nm['positive'], i, marker='D', color='black', s=80,
                       zorder=5, label='NucMed % positive' if i == 0 else '')

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df['dimension'])
    ax.set_xlabel('Percentage of Ratings')
    ax.set_title('ICU Rating Distribution with Nuclear Medicine Comparison\n'
                 '(Diamonds = Nuclear Medicine % positive)')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 105)

    plt.tight_layout()
    return fig


def create_figure_v3_difference_plot(comparison_df: pd.DataFrame) -> plt.Figure:
    """Dot plot showing difference in % positive (ICU - Nuclear Medicine)."""
    setup_plotting()

    plot_df = comparison_df.dropna(subset=['Difference_positive_pct']).copy()
    plot_df = plot_df.sort_values('Difference_positive_pct')

    fig, ax = plt.subplots(figsize=(8, 5))

    y = np.arange(len(plot_df))
    colors = [COLORS['success'] if d > 0 else COLORS['quaternary']
              for d in plot_df['Difference_positive_pct']]

    ax.barh(y, plot_df['Difference_positive_pct'], color=colors, alpha=0.8,
            edgecolor='white')

    # Add value labels
    for i, (val, dim) in enumerate(zip(plot_df['Difference_positive_pct'],
                                        plot_df['Dimension'])):
        offset = 1.5 if val >= 0 else -1.5
        ha = 'left' if val >= 0 else 'right'
        ax.text(val + offset, i, f'{val:+.1f}pp', va='center', ha=ha, fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df['Dimension'])
    ax.set_xlabel('Difference in % Positive Ratings (ICU − Nuclear Medicine, pp)')
    ax.set_title('Difference in Positive Rating Percentage:\nICU vs Nuclear Medicine')
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig


def create_figure_v4_radar(icu_pct: pd.DataFrame) -> plt.Figure:
    """Radar chart overlaying ICU and Nuclear Medicine % positive."""
    setup_plotting()

    # Only dimensions with nuclear medicine data
    dims_with_data = [d for d in icu_pct['dimension']
                      if NUCLEAR_MED_DATA.get(d, {}).get('positive') is not None]

    icu_vals = []
    nm_vals = []
    for dim in dims_with_data:
        icu_row = icu_pct[icu_pct['dimension'] == dim]
        icu_vals.append(icu_row['positive_pct'].values[0])
        nm_vals.append(NUCLEAR_MED_DATA[dim]['positive'])

    n = len(dims_with_data)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    icu_vals += icu_vals[:1]
    nm_vals += nm_vals[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.plot(angles, icu_vals, 'o-', linewidth=2, color=COLORS['primary'],
            label='ICU', markersize=8)
    ax.fill(angles, icu_vals, alpha=0.15, color=COLORS['primary'])

    ax.plot(angles, nm_vals, 's--', linewidth=2, color=COLORS['tertiary'],
            label='Nuclear Medicine', markersize=8)
    ax.fill(angles, nm_vals, alpha=0.15, color=COLORS['tertiary'])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims_with_data, size=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(['20%', '40%', '60%', '80%'])

    # Reference circle at 50%
    ref = [50] * (n + 1)
    ax.plot(angles, ref, '--', linewidth=1, color=COLORS['neutral'], alpha=0.5)

    ax.set_title('% Positive Ratings: ICU vs Nuclear Medicine\n'
                 'Meditron-3 (70B) Base Model', y=1.12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.05))

    plt.tight_layout()
    return fig


def main():
    """Main function for cross-specialty comparison."""

    print("=" * 60)
    print("08_cross_specialty_comparison.py - ICU vs Nuclear Medicine")
    print("=" * 60)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load ICU data
    print("\n1. Loading ICU data...")
    df = load_data()
    df_rated = get_rated_answers(df)
    print(f"   {len(df_rated)} ICU answer evaluations")

    # Compute ICU percentages
    print("\n2. Computing ICU percentage ratings...")
    icu_pct = compute_icu_percentages(df_rated)

    print("\n   ICU Rating Distribution:")
    for _, row in icu_pct.iterrows():
        print(f"   - {row['dimension']}: "
              f"{row['positive_pct']:.1f}% positive, "
              f"{row['neutral_pct']:.1f}% neutral, "
              f"{row['negative_pct']:.1f}% negative")

    # Build comparison table
    print("\n3. Building cross-specialty comparison...")
    comparison_df = build_comparison_table(icu_pct)

    # Print comparison for dimensions where nuclear medicine data exists
    print("\n   Cross-specialty comparison (% positive ratings):")
    print(f"   {'Dimension':<30} {'ICU':>8} {'NucMed':>8} {'Diff':>8}")
    print("   " + "-" * 56)
    for _, row in comparison_df.iterrows():
        nm_str = f"{row['NucMed_positive_pct']:.0f}%" if pd.notna(row['NucMed_positive_pct']) else "N/A"
        diff_str = f"{row['Difference_positive_pct']:+.1f}pp" if pd.notna(row['Difference_positive_pct']) else ""
        print(f"   {row['Dimension']:<30} {row['ICU_positive_pct']:>7.1f}% {nm_str:>8} {diff_str:>8}")

    # Save tables
    print("\n4. Saving tables...")
    comparison_df.to_csv(TABLES_DIR / '08_cross_specialty_comparison.csv', index=False)
    icu_pct.to_csv(TABLES_DIR / '08_icu_rating_distribution.csv', index=False)

    # Save metadata
    meta_df = pd.DataFrame([{
        'Field': 'ICU evaluations', 'Value': len(df_rated),
    }, {
        'Field': 'ICU questions', 'Value': df['question_id'].nunique(),
    }, {
        'Field': 'ICU raters', 'Value': df['Name'].nunique(),
    }, {
        'Field': 'NucMed evaluations', 'Value': NUCLEAR_MED_META['n_evaluations'],
    }, {
        'Field': 'NucMed questions', 'Value': NUCLEAR_MED_META['n_questions'],
    }, {
        'Field': 'NucMed raters', 'Value': NUCLEAR_MED_META['n_raters'],
    }, {
        'Field': 'NucMed reference', 'Value': NUCLEAR_MED_META['reference'],
    }, {
        'Field': 'Comparison metric', 'Value': '% positive (score >= 4)',
    }])
    meta_df.to_csv(TABLES_DIR / '08_cross_specialty_meta.csv', index=False)
    print(f"   Saved to: {TABLES_DIR}")

    # Create figures
    print("\n5. Creating figures...")

    fig1 = create_figure_v1_grouped_bar(comparison_df)
    save_figure_variants(fig1, '08_cross_specialty_comparison', FIGURES_DIR, 1)
    print("   - v1: Grouped bar chart")

    fig2 = create_figure_v2_stacked_bar(icu_pct)
    save_figure_variants(fig2, '08_cross_specialty_comparison', FIGURES_DIR, 2)
    print("   - v2: Stacked bar with NucMed overlay")

    fig3 = create_figure_v3_difference_plot(comparison_df)
    save_figure_variants(fig3, '08_cross_specialty_comparison', FIGURES_DIR, 3)
    print("   - v3: Difference plot")

    fig4 = create_figure_v4_radar(icu_pct)
    save_figure_variants(fig4, '08_cross_specialty_comparison', FIGURES_DIR, 4)
    print("   - v4: Radar chart")

    # Summary
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

    available = comparison_df.dropna(subset=['Difference_positive_pct'])
    if len(available) > 0:
        print(f"\nKey findings (ICU vs Nuclear Medicine, {len(available)} comparable dimensions):")
        mean_diff = available['Difference_positive_pct'].mean()
        print(f"  - Mean difference in % positive: {mean_diff:+.1f} percentage points")
        best = available.loc[available['Difference_positive_pct'].idxmax()]
        worst = available.loc[available['Difference_positive_pct'].idxmin()]
        print(f"  - Largest ICU advantage: {best['Dimension']} ({best['Difference_positive_pct']:+.1f}pp)")
        print(f"  - Largest NucMed advantage: {worst['Dimension']} ({worst['Difference_positive_pct']:+.1f}pp)")


if __name__ == '__main__':
    main()
