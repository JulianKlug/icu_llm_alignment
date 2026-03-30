#!/usr/bin/env python3
"""
09_clinical_error_analysis.py
==============================
Supplementary Analysis: Clinical error analysis and tail risk reporting.

Computes:
1. Tail risk distribution (% of ratings scoring <=2 per dimension)
2. Low-scoring answer categorization by error type
3. Illustrative question-answer-evaluation triads

Output:
- output/tables/09_tail_risk_distribution.csv
- output/tables/09_low_scoring_answers.csv
- output/tables/09_error_category_summary.csv
- output/tables/09_illustrative_examples.csv
- output/figures/09_clinical_error_analysis_v[1-4].png
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
    load_data, create_concatenated_answers_df,
    setup_plotting, save_figure_variants, COLORS, PALETTE,
    DIMENSION_NAMES, EVAL_COLS
)

OUTPUT_DIR = project_root / 'output'
TABLES_DIR = OUTPUT_DIR / 'tables'
FIGURES_DIR = OUTPUT_DIR / 'figures'

# Clinically critical dimensions for error categorization
CRITICAL_DIMS = [
    'Eval Alignment with Guidelines',
    'Eval Harmlessness',
    'Eval Relevance & Completeness',
]


def _eval_col_to_dim_name(eval_col: str) -> str:
    """Map EVAL_COLS name to DIMENSION_NAMES name."""
    mapping = dict(zip(EVAL_COLS, DIMENSION_NAMES))
    return mapping.get(eval_col, eval_col.replace('Eval ', ''))


def compute_tail_risk(all_answers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute tail risk distribution for each dimension.

    Returns % of individual ratings scoring <=2, ==1, and full score
    distribution (1-5) as percentages.
    """
    results = []
    for domain in EVAL_COLS:
        dim_name = _eval_col_to_dim_name(domain)
        values = all_answers_df[domain].dropna()
        n = len(values)
        if n == 0:
            continue

        row = {
            'dimension': dim_name,
            'n_ratings': n,
            'pct_score_1': round(100 * (values == 1).sum() / n, 1),
            'pct_score_2': round(100 * (values == 2).sum() / n, 1),
            'pct_score_3': round(100 * (values == 3).sum() / n, 1),
            'pct_score_4': round(100 * (values == 4).sum() / n, 1),
            'pct_score_5': round(100 * (values == 5).sum() / n, 1),
            'pct_leq2': round(100 * (values <= 2).sum() / n, 1),
            'n_leq2': int((values <= 2).sum()),
            'pct_eq1': round(100 * (values == 1).sum() / n, 1),
            'n_eq1': int((values == 1).sum()),
        }
        results.append(row)

    df = pd.DataFrame(results)
    df = df.sort_values('pct_leq2', ascending=False).reset_index(drop=True)
    return df


def compute_answer_level_risk(all_answers_df: pd.DataFrame) -> dict:
    """
    Compute aggregate answer-level tail risk statistics.

    Returns dict with summary stats about how many answers have at least
    one dimension scoring <=2.
    """
    # Group by answer and compute per-answer min score across dimensions
    answer_groups = all_answers_df.groupby('Answer')

    n_answers = all_answers_df['Answer'].nunique()
    answers_with_any_leq2 = 0
    answers_with_any_eq1 = 0

    for _, group in answer_groups:
        scores = group[EVAL_COLS].values.flatten()
        scores = scores[~np.isnan(scores)]
        if len(scores) > 0:
            if (scores <= 2).any():
                answers_with_any_leq2 += 1
            if (scores == 1).any():
                answers_with_any_eq1 += 1

    return {
        'n_answers': n_answers,
        'n_with_any_leq2': answers_with_any_leq2,
        'pct_with_any_leq2': round(100 * answers_with_any_leq2 / n_answers, 1),
        'n_with_any_eq1': answers_with_any_eq1,
        'pct_with_any_eq1': round(100 * answers_with_any_eq1 / n_answers, 1),
    }


def identify_low_scoring_answers(all_answers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify answers where the mean score across raters is <=2 on any
    clinically critical dimension (Harmlessness, Guideline Alignment,
    Relevance & Completeness).

    Returns DataFrame with one row per flagged answer.
    """
    # Compute per-answer mean scores across raters
    answer_stats = []
    for answer in all_answers_df['Answer'].unique():
        adf = all_answers_df[all_answers_df['Answer'] == answer]
        question = adf['Question'].iloc[0]
        question_id = adf['question_id'].iloc[0]
        answer_type = adf['answer_type'].iloc[0]
        n_raters = len(adf)

        row = {
            'question_id': question_id,
            'Question': question,
            'Answer': answer,
            'answer_type': answer_type,
            'n_raters': n_raters,
        }

        for domain in EVAL_COLS:
            dim_name = _eval_col_to_dim_name(domain)
            vals = adf[domain].dropna()
            row[f'{dim_name}_mean'] = vals.mean() if len(vals) > 0 else np.nan

        answer_stats.append(row)

    stats_df = pd.DataFrame(answer_stats)

    # Flag answers with mean <=2 on any critical dimension
    critical_dim_names = [d.replace('Eval ', '') for d in CRITICAL_DIMS]
    critical_mean_cols = [f'{d}_mean' for d in critical_dim_names]

    mask = pd.Series(False, index=stats_df.index)
    for col in critical_mean_cols:
        mask = mask | (stats_df[col] <= 2.0)

    flagged = stats_df[mask].copy()

    # Categorize error type
    categories = []
    for _, row in flagged.iterrows():
        cats = []
        if row.get('Alignment with Guidelines_mean', 5) <= 2.0:
            cats.append('Low guideline alignment')
        if row.get('Harmlessness_mean', 5) <= 2.0:
            cats.append('Safety concern')
        if row.get('Relevance & Completeness_mean', 5) <= 2.0:
            cats.append('Incomplete/irrelevant')

        if len(cats) >= 2:
            categories.append('Multiple concerns')
        elif len(cats) == 1:
            categories.append(cats[0])
        else:
            categories.append('Other')

    flagged['error_category'] = categories

    # Truncate answer text for readability
    flagged['Answer_truncated'] = flagged['Answer'].str[:500]

    # Sort by worst critical dimension score
    flagged['worst_critical_mean'] = flagged[critical_mean_cols].min(axis=1)
    flagged = flagged.sort_values('worst_critical_mean').reset_index(drop=True)

    return flagged


def summarize_error_categories(flagged_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize error categories."""
    if len(flagged_df) == 0:
        return pd.DataFrame(columns=['error_category', 'n', 'pct'])

    summary = flagged_df['error_category'].value_counts().reset_index()
    summary.columns = ['error_category', 'n']
    summary['pct'] = round(100 * summary['n'] / summary['n'].sum(), 1)
    return summary


def select_illustrative_examples(all_answers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Select 3 illustrative examples: good, mediocre, concerning.

    Selection is based on mean score across all dimensions averaged over raters.
    """
    # Compute per-answer overall mean
    answer_stats = []
    for answer in all_answers_df['Answer'].unique():
        adf = all_answers_df[all_answers_df['Answer'] == answer]
        question = adf['Question'].iloc[0]
        question_id = adf['question_id'].iloc[0]
        answer_type = adf['answer_type'].iloc[0]
        n_raters = len(adf)

        row = {
            'question_id': question_id,
            'Question': question,
            'Answer': answer,
            'answer_type': answer_type,
            'n_raters': n_raters,
        }

        dim_means = []
        for domain in EVAL_COLS:
            dim_name = _eval_col_to_dim_name(domain)
            vals = adf[domain].dropna()
            m = vals.mean() if len(vals) > 0 else np.nan
            row[f'{dim_name}_mean'] = m
            if not np.isnan(m):
                dim_means.append(m)

        row['overall_mean'] = np.mean(dim_means) if dim_means else np.nan
        answer_stats.append(row)

    stats_df = pd.DataFrame(answer_stats).dropna(subset=['overall_mean'])

    # Select examples (require >=2 raters for stability)
    stats_df = stats_df[stats_df['n_raters'] >= 2]

    # Good: highest overall mean
    good_idx = stats_df['overall_mean'].idxmax()

    # Concerning: lowest Harmlessness or Guideline Alignment mean
    harm_col = 'Harmlessness_mean'
    align_col = 'Alignment with Guidelines_mean'
    stats_df['worst_safety'] = stats_df[[harm_col, align_col]].min(axis=1)
    concerning_idx = stats_df['worst_safety'].idxmin()

    # Mediocre: closest to the grand mean, excluding already selected
    grand_mean = stats_df['overall_mean'].mean()
    remaining = stats_df.drop(index=[good_idx, concerning_idx])
    mediocre_idx = (remaining['overall_mean'] - grand_mean).abs().idxmin()

    examples = stats_df.loc[[good_idx, mediocre_idx, concerning_idx]].copy()
    examples['example_type'] = ['Good', 'Mediocre', 'Concerning']

    # Truncate text
    examples['Answer_truncated'] = examples['Answer'].str[:500]
    examples['Question_truncated'] = examples['Question'].str[:300]

    return examples


def create_figure_v1_tail_risk_bar(tail_risk_df: pd.DataFrame) -> plt.Figure:
    """Horizontal bar chart of % negative (<=2) per dimension."""
    setup_plotting()

    fig, ax = plt.subplots(figsize=(10, 6))

    df = tail_risk_df.sort_values('pct_leq2', ascending=True)

    y = np.arange(len(df))
    bars = ax.barh(y, df['pct_leq2'], color=COLORS['quaternary'], alpha=0.8)

    # Add score=1 portion as darker segment
    ax.barh(y, df['pct_eq1'], color='#8B0000', alpha=0.9, label='Score = 1')

    # Value labels
    for i, (pct, n) in enumerate(zip(df['pct_leq2'], df['n_leq2'])):
        ax.text(pct + 0.5, i, f'{pct:.1f}% (n={n})', va='center', fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(df['dimension'])
    ax.set_xlabel('% of Ratings Scoring ≤ 2')
    ax.set_title('Tail Risk: Proportion of Negative Ratings per Dimension\n'
                 '(Dark red = score 1, light red = score 2)')
    ax.legend(loc='lower right')
    ax.set_xlim(0, max(df['pct_leq2']) + 8)

    plt.tight_layout()
    return fig


def create_figure_v2_stacked_distribution(tail_risk_df: pd.DataFrame) -> plt.Figure:
    """Stacked bar showing full score distribution (1-5) per dimension."""
    setup_plotting()

    df = tail_risk_df.sort_values('pct_leq2', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 7))

    y = np.arange(len(df))
    score_colors = {
        1: '#c0392b',   # Dark red
        2: '#e74c3c',   # Red
        3: '#f39c12',   # Orange/neutral
        4: '#27ae60',   # Green
        5: '#1a7a3a',   # Dark green
    }

    left = np.zeros(len(df))
    for score in [1, 2, 3, 4, 5]:
        col = f'pct_score_{score}'
        widths = df[col].values
        ax.barh(y, widths, left=left, color=score_colors[score],
                label=f'Score {score}', alpha=0.85)
        left += widths

    ax.set_yticks(y)
    ax.set_yticklabels(df['dimension'])
    ax.set_xlabel('Percentage of Ratings')
    ax.set_title('Score Distribution by Evaluation Dimension\n'
                 '(Red = scores 1-2, Orange = score 3, Green = scores 4-5)')
    ax.legend(loc='lower right', ncol=5, fontsize=9)
    ax.set_xlim(0, 105)

    plt.tight_layout()
    return fig


def create_figure_v3_heatmap(flagged_df: pd.DataFrame) -> plt.Figure:
    """Heatmap of low-scoring answers (answers x dimensions)."""
    setup_plotting()

    if len(flagged_df) == 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, 'No answers with mean score ≤ 2\non critical dimensions',
                ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig

    dim_mean_cols = [f'{d}_mean' for d in DIMENSION_NAMES]
    available_cols = [c for c in dim_mean_cols if c in flagged_df.columns]

    # Limit to top 25 worst answers for readability
    plot_df = flagged_df.head(25)

    heatmap_data = plot_df[available_cols].values
    dim_labels = [c.replace('_mean', '') for c in available_cols]

    # Row labels: truncated question + error category
    row_labels = [f"Q{row['question_id']} ({row['error_category'][:15]})"
                  for _, row in plot_df.iterrows()]

    fig, ax = plt.subplots(figsize=(14, max(6, len(plot_df) * 0.4)))

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        vmin=1,
        vmax=5,
        xticklabels=dim_labels,
        yticklabels=row_labels,
        ax=ax,
        cbar_kws={'label': 'Mean Score (across raters)'},
        linewidths=0.5,
    )

    ax.set_title(f'Low-Scoring Answers: Mean Ratings by Dimension\n'
                 f'(n = {len(plot_df)} answers with mean ≤ 2 on a critical dimension)')
    ax.set_xlabel('Evaluation Dimension')
    ax.set_ylabel('Answer (Question ID, Error Category)')

    plt.tight_layout()
    return fig


def create_figure_v4_radar_examples(examples_df: pd.DataFrame) -> plt.Figure:
    """Radar chart comparing the 3 illustrative examples."""
    setup_plotting()

    dim_mean_cols = [f'{d}_mean' for d in DIMENSION_NAMES]
    available_cols = [c for c in dim_mean_cols if c in examples_df.columns]
    dim_labels = [c.replace('_mean', '') for c in available_cols]

    n = len(dim_labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    example_colors = {
        'Good': COLORS['success'],
        'Mediocre': COLORS['tertiary'],
        'Concerning': COLORS['quaternary'],
    }

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for _, row in examples_df.iterrows():
        etype = row['example_type']
        values = [row[c] for c in available_cols]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=etype,
                color=example_colors.get(etype, COLORS['neutral']), markersize=8)
        ax.fill(angles, values, alpha=0.1,
                color=example_colors.get(etype, COLORS['neutral']))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, size=8)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])

    # Reference at 3
    ref = [3] * (n + 1)
    ax.plot(angles, ref, '--', linewidth=1, color=COLORS['neutral'], alpha=0.5)

    ax.set_title('Illustrative Examples: Performance Profiles\n'
                 '(Good / Mediocre / Concerning)', y=1.12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.05))

    plt.tight_layout()
    return fig


def main():
    """Main function for clinical error analysis."""

    print("=" * 60)
    print("09_clinical_error_analysis.py - Clinical Error Analysis")
    print("=" * 60)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n1. Loading data...")
    df = load_data()
    all_answers_df = create_concatenated_answers_df(df)
    print(f"   {len(all_answers_df)} answer-rater evaluations")
    print(f"   {all_answers_df['Answer'].nunique()} unique answers")

    # Tail risk
    print("\n2. Computing tail risk distribution...")
    tail_risk_df = compute_tail_risk(all_answers_df)
    print("\n   % of ratings scoring ≤ 2 per dimension:")
    for _, row in tail_risk_df.iterrows():
        print(f"   - {row['dimension']}: {row['pct_leq2']:.1f}% "
              f"(n={row['n_leq2']}, of which {row['pct_eq1']:.1f}% score=1)")

    # Answer-level risk
    answer_risk = compute_answer_level_risk(all_answers_df)
    print(f"\n   Answer-level risk:")
    print(f"   - {answer_risk['pct_with_any_leq2']:.1f}% of answers have ≥1 rating ≤ 2 "
          f"({answer_risk['n_with_any_leq2']}/{answer_risk['n_answers']})")
    print(f"   - {answer_risk['pct_with_any_eq1']:.1f}% of answers have ≥1 rating = 1 "
          f"({answer_risk['n_with_any_eq1']}/{answer_risk['n_answers']})")

    # Low-scoring answers
    print("\n3. Identifying low-scoring answers...")
    flagged_df = identify_low_scoring_answers(all_answers_df)
    print(f"   {len(flagged_df)} answers flagged (mean ≤ 2 on a critical dimension)")

    error_summary = summarize_error_categories(flagged_df)
    if len(error_summary) > 0:
        print("\n   Error categories:")
        for _, row in error_summary.iterrows():
            print(f"   - {row['error_category']}: n={row['n']} ({row['pct']:.1f}%)")

    # Illustrative examples
    print("\n4. Selecting illustrative examples...")
    examples_df = select_illustrative_examples(all_answers_df)
    for _, row in examples_df.iterrows():
        print(f"\n   {row['example_type']} example (overall mean={row['overall_mean']:.2f}):")
        print(f"   Q: {row['Question'][:100]}...")

    # Save tables
    print("\n5. Saving tables...")
    tail_risk_df.to_csv(TABLES_DIR / '09_tail_risk_distribution.csv', index=False)

    # Save flagged answers (without full answer text for CSV readability)
    if len(flagged_df) > 0:
        save_cols = ['question_id', 'answer_type', 'n_raters', 'error_category',
                     'Answer_truncated', 'Question']
        dim_mean_cols = [f'{d}_mean' for d in DIMENSION_NAMES]
        save_cols += [c for c in dim_mean_cols if c in flagged_df.columns]
        flagged_df[save_cols].to_csv(
            TABLES_DIR / '09_low_scoring_answers.csv', index=False)

    error_summary.to_csv(TABLES_DIR / '09_error_category_summary.csv', index=False)

    # Save illustrative examples
    save_cols_ex = ['example_type', 'question_id', 'n_raters', 'overall_mean',
                    'Question_truncated', 'Answer_truncated']
    dim_mean_cols = [f'{d}_mean' for d in DIMENSION_NAMES]
    save_cols_ex += [c for c in dim_mean_cols if c in examples_df.columns]
    examples_df[save_cols_ex].to_csv(
        TABLES_DIR / '09_illustrative_examples.csv', index=False)

    # Save answer-level risk as small table
    pd.DataFrame([answer_risk]).to_csv(
        TABLES_DIR / '09_answer_level_risk.csv', index=False)

    print(f"   Saved to: {TABLES_DIR}")

    # Create figures
    print("\n6. Creating figures...")

    fig1 = create_figure_v1_tail_risk_bar(tail_risk_df)
    save_figure_variants(fig1, '09_clinical_error_analysis', FIGURES_DIR, 1)
    print("   - v1: Tail risk bar chart")

    fig2 = create_figure_v2_stacked_distribution(tail_risk_df)
    save_figure_variants(fig2, '09_clinical_error_analysis', FIGURES_DIR, 2)
    print("   - v2: Stacked score distribution")

    fig3 = create_figure_v3_heatmap(flagged_df)
    save_figure_variants(fig3, '09_clinical_error_analysis', FIGURES_DIR, 3)
    print("   - v3: Low-scoring answers heatmap")

    fig4 = create_figure_v4_radar_examples(examples_df)
    save_figure_variants(fig4, '09_clinical_error_analysis', FIGURES_DIR, 4)
    print("   - v4: Illustrative examples radar")

    # Summary
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    print(f"\nKey findings:")
    worst = tail_risk_df.iloc[0]
    print(f"  - Highest tail risk: {worst['dimension']} "
          f"({worst['pct_leq2']:.1f}% of ratings ≤ 2)")
    print(f"  - {answer_risk['pct_with_any_leq2']:.1f}% of answers received "
          f"at least one rating ≤ 2")
    print(f"  - {len(flagged_df)} answers flagged with mean ≤ 2 "
          f"on a critical dimension")


if __name__ == '__main__':
    main()
