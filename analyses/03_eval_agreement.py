#!/usr/bin/env python3
"""
03_eval_agreement.py
====================
Analysis 2: Interrater agreement for performance on each evaluation domain.

This script computes agreement metrics for each evaluation dimension by:
1. Concatenating all answers (first + second) into one dataframe
2. Grouping by unique Answer text
3. Computing std and Krippendorff's alpha per answer per domain
4. Summarizing agreement across all answers

Output:
- output/tables/03_eval_agreement.csv (summary per domain)
- output/tables/03_eval_agreement_per_answer.csv (per-answer metrics)
- output/figures/03_eval_agreement_v[1-4].png
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
import krippendorff

from analyses.utils import (
    load_data, create_concatenated_answers_df,
    setup_plotting, save_figure_variants, COLORS, PALETTE,
    DIMENSION_NAMES, FIRST_EVAL_COLS, SECOND_EVAL_COLS, EVAL_COLS
)

OUTPUT_DIR = project_root / 'output'
TABLES_DIR = OUTPUT_DIR / 'tables'
FIGURES_DIR = OUTPUT_DIR / 'figures'


def compute_std_per_answer(all_answers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute standard deviation of ratings per answer per domain.

    Returns:
        DataFrame with per-answer std for each domain
    """
    std_rows = []

    unique_answers = all_answers_df['Answer'].unique()

    for answer in unique_answers:
        answer_df = all_answers_df[all_answers_df['Answer'] == answer]
        question = answer_df['Question'].iloc[0]
        n_raters = len(answer_df)

        row_std = {'Answer': answer, 'Question': question, 'n_raters': n_raters}

        for domain in EVAL_COLS:
            # Get ratings for this domain
            data = answer_df[domain].values
            valid_data = data[~pd.isna(data)]

            # Compute std
            if len(valid_data) >= 2:
                domain_std = np.std(valid_data, ddof=1)  # Sample std
            else:
                domain_std = np.nan
            row_std[domain] = domain_std

        std_rows.append(row_std)

    return pd.DataFrame(std_rows)


def compute_alpha_per_dimension(all_answers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Krippendorff's alpha per dimension across all answers.

    Creates a proper items-by-raters reliability matrix for each dimension,
    where items are unique answers and raters are columns.

    Returns:
        DataFrame with one row per dimension containing alpha and metadata
    """
    alpha_rows = []

    for domain in EVAL_COLS:
        dim_name = domain.replace('Eval ', '')

        # Build items-by-raters matrix: rows = answers, columns = raters
        pivot = all_answers_df.pivot_table(
            index='Answer',
            columns='Name',
            values=domain,
            aggfunc='first'
        )

        # Filter to answers with at least 2 raters
        n_raters_per_answer = pivot.notna().sum(axis=1)
        pivot = pivot[n_raters_per_answer >= 2]

        n_items = len(pivot)
        n_raters = pivot.shape[1]

        if n_items < 2:
            alpha_val = np.nan
        else:
            try:
                # Clip any out-of-range values (e.g., 0) to valid Likert range
                pivot_clipped = pivot.clip(lower=1, upper=5)
                # krippendorff expects reliability_data as raters x items
                reliability_data = pivot_clipped.values.T
                alpha_val = krippendorff.alpha(
                    reliability_data=reliability_data,
                    level_of_measurement='ordinal',
                    value_domain=[1, 2, 3, 4, 5]
                )
            except Exception:
                alpha_val = np.nan

        alpha_rows.append({
            'dimension': dim_name,
            'alpha': alpha_val,
            'n_items': n_items,
            'n_raters': n_raters,
        })

    return pd.DataFrame(alpha_rows)


def summarize_agreement(std_df: pd.DataFrame, alpha_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary statistics of agreement across all answers.

    Args:
        std_df: Per-answer std DataFrame
        alpha_df: Per-dimension Krippendorff's alpha DataFrame
    """
    summary_rows = []

    for domain in EVAL_COLS:
        # Get short dimension name
        dim_name = domain.replace('Eval ', '')

        # Std statistics
        std_values = std_df[domain].dropna()

        # Get per-dimension alpha from the alpha_df
        alpha_row = alpha_df[alpha_df['dimension'] == dim_name]
        alpha_val = alpha_row['alpha'].values[0] if len(alpha_row) > 0 else np.nan

        summary_rows.append({
            'dimension': dim_name,
            'n_answers': len(std_values),
            'std_mean': std_values.mean(),
            'std_median': std_values.median(),
            'std_sd': std_values.std(),
            'std_min': std_values.min(),
            'std_max': std_values.max(),
            'krippendorff_alpha': alpha_val,
            'pct_perfect_agreement': (std_values == 0).mean() * 100,
        })

    return pd.DataFrame(summary_rows)


def create_figure_v1_std_boxplot(std_df: pd.DataFrame) -> plt.Figure:
    """Create boxplot of std distribution per domain."""
    setup_plotting()

    fig, ax = plt.subplots(figsize=(12, 6))

    data = [std_df[col].dropna().values for col in EVAL_COLS]
    labels = [col.replace('Eval ', '') for col in EVAL_COLS]

    bp = ax.boxplot(data, patch_artist=True)

    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(PALETTE[i % len(PALETTE)])
        patch.set_alpha(0.7)

    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Standard Deviation of Ratings')
    ax.set_xlabel('Evaluation Dimension')
    ax.set_title('Interrater Variability per Dimension\n(Lower = Better Agreement)')
    ax.axhline(y=0, color=COLORS['success'], linestyle='--', alpha=0.5, label='Perfect agreement')
    ax.legend()

    plt.tight_layout()
    return fig


def create_figure_v2_std_bar(summary_df: pd.DataFrame) -> plt.Figure:
    """Create bar chart of mean std per domain."""
    setup_plotting()

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(summary_df))
    means = summary_df['std_mean'].values
    stds = summary_df['std_sd'].values

    # Color by agreement level (lower std = better = greener)
    colors = []
    for mean in means:
        if mean < 0.5:
            colors.append('#4575b4')  # Good
        elif mean < 0.75:
            colors.append('#91bfdb')  # Moderate
        elif mean < 1.0:
            colors.append('#fee090')  # Fair
        else:
            colors.append('#d73027')  # Poor

    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.8, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['dimension'].values, rotation=45, ha='right')
    ax.set_ylabel('Mean Standard Deviation (± SD)')
    ax.set_xlabel('Evaluation Dimension')
    ax.set_title('Mean Rating Variability per Dimension')

    # Add value labels
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def create_figure_v3_agreement_heatmap(std_df: pd.DataFrame) -> plt.Figure:
    """Create heatmap showing std distribution."""
    setup_plotting()

    fig, ax = plt.subplots(figsize=(12, 8))

    # Compute mean std per domain
    means = [std_df[col].mean() for col in EVAL_COLS]
    medians = [std_df[col].median() for col in EVAL_COLS]
    pct_zero = [(std_df[col] == 0).mean() * 100 for col in EVAL_COLS]

    data = np.array([means, medians, pct_zero])

    labels = [col.replace('Eval ', '') for col in EVAL_COLS]
    row_labels = ['Mean Std', 'Median Std', '% Perfect Agreement']

    sns.heatmap(
        data,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn_r',
        xticklabels=labels,
        yticklabels=row_labels,
        ax=ax
    )

    ax.set_title('Agreement Summary per Dimension')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig


def create_figure_v4_distribution(std_df: pd.DataFrame) -> plt.Figure:
    """Create violin plots showing full distribution of std per domain."""
    setup_plotting()

    fig, ax = plt.subplots(figsize=(14, 7))

    data = [std_df[col].dropna().values for col in EVAL_COLS]
    labels = [col.replace('Eval ', '')[:15] for col in EVAL_COLS]

    parts = ax.violinplot(data, positions=range(len(EVAL_COLS)), showmeans=True, showmedians=True)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(PALETTE[i % len(PALETTE)])
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(EVAL_COLS)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Standard Deviation of Ratings')
    ax.set_xlabel('Evaluation Dimension')
    ax.set_title('Distribution of Rating Variability per Answer')
    ax.axhline(y=0, color=COLORS['success'], linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig


def main():
    """Main function for evaluation agreement analysis."""

    print("=" * 60)
    print("03_eval_agreement.py - Interrater Agreement per Dimension")
    print("=" * 60)

    # Create output directories
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n1. Loading data...")
    df = load_data()
    print(f"   Loaded {len(df)} ratings")

    # Create concatenated answers dataframe
    print("\n2. Concatenating all answers...")
    all_answers_df = create_concatenated_answers_df(df)

    n_first = len(df)
    n_second = df['Second Answer'].notna().sum()
    n_unique_answers = all_answers_df['Answer'].nunique()

    print(f"   Total answer-rater pairs: {len(all_answers_df)}")
    print(f"   Unique answers: {n_unique_answers}")

    # Compute std per answer
    print("\n3. Computing std per answer...")
    std_df = compute_std_per_answer(all_answers_df)
    print(f"   Computed std for {len(std_df)} answers")

    # Compute Krippendorff's alpha per dimension (proper multi-item computation)
    print("\n4. Computing Krippendorff's alpha per dimension...")
    alpha_df = compute_alpha_per_dimension(all_answers_df)
    for _, row in alpha_df.iterrows():
        print(f"   - {row['dimension']}: alpha={row['alpha']:.3f} "
              f"(n_items={row['n_items']}, n_raters={row['n_raters']})")

    # Summarize agreement
    print("\n5. Summarizing agreement across answers...")
    summary_df = summarize_agreement(std_df, alpha_df)

    print("\n   Results (Mean Std per Dimension):")
    for _, row in summary_df.iterrows():
        print(f"   - {row['dimension']}: std={row['std_mean']:.3f}, "
              f"alpha={row['krippendorff_alpha']:.3f}, "
              f"perfect_agreement={row['pct_perfect_agreement']:.1f}%")

    # Save tables
    print("\n6. Saving tables...")
    summary_df.to_csv(TABLES_DIR / '03_eval_agreement.csv', index=False)
    std_df.to_csv(TABLES_DIR / '03_eval_agreement_std_per_answer.csv', index=False)
    alpha_df.to_csv(TABLES_DIR / '03_eval_agreement_alpha_per_dimension.csv', index=False)
    print(f"   Saved to: {TABLES_DIR}")

    # Create figures
    print("\n7. Creating figures...")

    fig1 = create_figure_v1_std_boxplot(std_df)
    save_figure_variants(fig1, '03_eval_agreement', FIGURES_DIR, 1)
    print("   - Saved: Std boxplot (v1)")

    fig2 = create_figure_v2_std_bar(summary_df)
    save_figure_variants(fig2, '03_eval_agreement', FIGURES_DIR, 2)
    print("   - Saved: Mean std bar chart (v2)")

    fig3 = create_figure_v3_agreement_heatmap(std_df)
    save_figure_variants(fig3, '03_eval_agreement', FIGURES_DIR, 3)
    print("   - Saved: Agreement heatmap (v3)")

    fig4 = create_figure_v4_distribution(std_df)
    save_figure_variants(fig4, '03_eval_agreement', FIGURES_DIR, 4)
    print("   - Saved: Distribution violin plot (v4)")

    # Summary
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

    best_dim = summary_df.loc[summary_df['std_mean'].idxmin()]
    worst_dim = summary_df.loc[summary_df['std_mean'].idxmax()]

    print(f"\nKey findings (across {n_unique_answers} unique answers):")
    print(f"  - Best agreement (lowest std): {best_dim['dimension']} (std={best_dim['std_mean']:.3f})")
    print(f"  - Worst agreement (highest std): {worst_dim['dimension']} (std={worst_dim['std_mean']:.3f})")
    print(f"  - Overall mean std: {summary_df['std_mean'].mean():.3f}")


if __name__ == '__main__':
    main()
