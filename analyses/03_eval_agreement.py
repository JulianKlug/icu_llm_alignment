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
    load_data, setup_plotting, save_figure_variants, COLORS, PALETTE,
    DIMENSION_NAMES, FIRST_EVAL_COLS, SECOND_EVAL_COLS
)

OUTPUT_DIR = project_root / 'output'
TABLES_DIR = OUTPUT_DIR / 'tables'
FIGURES_DIR = OUTPUT_DIR / 'figures'

# Standardized eval column names
EVAL_COLS = [
    'Eval Alignment with Guidelines', 'Eval Question Comprehension',
    'Eval Logical Reasoning', 'Eval Relevance & Completeness',
    'Eval Harmlessness', 'Eval Fairness', 'Eval Contextual Awareness',
    'Eval Your Confidence', 'Eval Model Confidence',
    'Eval Communication & Clarity'
]


def create_concatenated_answers_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate first and second answers into one dataframe with standardized columns.

    Returns:
        DataFrame with columns: Answer, Question, Name, and standardized Eval columns
    """
    # First answer dataframe
    first_answer_df = df[['First Answer', 'Question', 'Name'] + FIRST_EVAL_COLS].copy()
    first_answer_df = first_answer_df.rename(columns={'First Answer': 'Answer'})

    # Second answer dataframe - rename columns to match first answer
    second_cols_map = dict(zip(SECOND_EVAL_COLS, EVAL_COLS))
    second_cols_map['Second Answer'] = 'Answer'

    second_answer_df = df[['Second Answer', 'Question', 'Name'] + SECOND_EVAL_COLS].copy()
    second_answer_df = second_answer_df.rename(columns=second_cols_map)

    # Concatenate
    all_answers_df = pd.concat([first_answer_df, second_answer_df], ignore_index=True)

    # Remove rows where Answer is NaN
    all_answers_df = all_answers_df.dropna(subset=['Answer'])

    return all_answers_df


def compute_agreement_per_answer(all_answers_df: pd.DataFrame) -> tuple:
    """
    Compute agreement metrics (std and Krippendorff's alpha) per answer per domain.

    Returns:
        Tuple of (std_df, alpha_df) - DataFrames with per-answer agreement metrics
    """
    std_rows = []
    alpha_rows = []

    unique_answers = all_answers_df['Answer'].unique()

    for answer in unique_answers:
        answer_df = all_answers_df[all_answers_df['Answer'] == answer]
        question = answer_df['Question'].iloc[0]
        n_raters = len(answer_df)

        row_std = {'Answer': answer, 'Question': question, 'n_raters': n_raters}
        row_alpha = {'Answer': answer, 'Question': question, 'n_raters': n_raters}

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

            # Compute Krippendorff's alpha
            if len(valid_data) <= 1:
                domain_alpha = np.nan
            elif len(set(valid_data)) == 1:
                # All values are the same - perfect agreement
                domain_alpha = 1.0
            else:
                try:
                    # Reshape for krippendorff: each row is a rater, each column is an item
                    # Here we have one item (the answer) and multiple raters
                    domain_alpha = krippendorff.alpha(
                        reliability_data=[valid_data],
                        level_of_measurement='ordinal',
                        value_domain=[0, 1, 2, 3, 4, 5]
                    )
                except:
                    domain_alpha = np.nan
            row_alpha[domain] = domain_alpha

        std_rows.append(row_std)
        alpha_rows.append(row_alpha)

    std_df = pd.DataFrame(std_rows)
    alpha_df = pd.DataFrame(alpha_rows)

    return std_df, alpha_df


def summarize_agreement(std_df: pd.DataFrame, alpha_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary statistics of agreement across all answers.
    """
    summary_rows = []

    for domain in EVAL_COLS:
        # Get short dimension name
        dim_name = domain.replace('Eval ', '')

        # Std statistics
        std_values = std_df[domain].dropna()

        # Alpha statistics
        alpha_values = alpha_df[domain].dropna()

        summary_rows.append({
            'dimension': dim_name,
            'n_answers': len(std_values),
            'std_mean': std_values.mean(),
            'std_median': std_values.median(),
            'std_sd': std_values.std(),
            'std_min': std_values.min(),
            'std_max': std_values.max(),
            'alpha_mean': alpha_values.mean() if len(alpha_values) > 0 else np.nan,
            'alpha_median': alpha_values.median() if len(alpha_values) > 0 else np.nan,
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

    # Compute agreement per answer
    print("\n3. Computing agreement metrics per answer...")
    std_df, alpha_df = compute_agreement_per_answer(all_answers_df)
    print(f"   Computed std and Krippendorff's alpha for {len(std_df)} answers")

    # Summarize agreement
    print("\n4. Summarizing agreement across answers...")
    summary_df = summarize_agreement(std_df, alpha_df)

    print("\n   Results (Mean Std per Dimension):")
    for _, row in summary_df.iterrows():
        print(f"   - {row['dimension']}: std={row['std_mean']:.3f}, "
              f"perfect_agreement={row['pct_perfect_agreement']:.1f}%")

    # Save tables
    print("\n5. Saving tables...")
    summary_df.to_csv(TABLES_DIR / '03_eval_agreement.csv', index=False)
    std_df.to_csv(TABLES_DIR / '03_eval_agreement_std_per_answer.csv', index=False)
    alpha_df.to_csv(TABLES_DIR / '03_eval_agreement_alpha_per_answer.csv', index=False)
    print(f"   Saved to: {TABLES_DIR}")

    # Create figures
    print("\n6. Creating figures...")

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
