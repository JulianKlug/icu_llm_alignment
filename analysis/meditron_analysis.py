#!/usr/bin/env python3
"""
Meditron ICU Evaluation Analysis

Analyzes human expert ratings of Meditron LLM responses to ICU-related questions.
Generates Table 1, main metrics, and visualization plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = Path('/mnt/data1/klug/datasets/meditron/Meditron_ICU.xlsx')
OUTPUT_DIR = Path('/home/klug/icu_projects/icu_llm_alignment/output')
OUTPUT_DIR.mkdir(exist_ok=True)

# Evaluation dimension columns
EVAL_COLS = [
    'Eval Alignment with Guidelines',
    'Eval Question Comprehension',
    'Eval Logical Reasoning',
    'Eval Relevance & Completeness',
    'Eval Harmlessness',
    'Eval Fairness',
    'Eval Contextual Awareness',
    'Eval Your Confidence',
    'Eval Model Confidence',
    'Eval Communication & Clarity'
]

SECOND_EVAL_COLS = [
    'Second Eval Alignment with Guidelines',
    'Second Eval Question Comprehension',
    'Second Eval Logical Reasoning',
    'Second Eval Relevance & Completeness',
    'Second Eval Harmlessness',
    'Second Eval Fairness',
    'Second Eval Contextual Awareness',
    'Second Eval Your Confidence',
    'Second Eval Model Confidence',
    'Second Eval Communication & Clarity'
]

# Short names for display
DIMENSION_NAMES = [
    'Alignment with Guidelines',
    'Question Comprehension',
    'Logical Reasoning',
    'Relevance & Completeness',
    'Harmlessness',
    'Fairness',
    'Contextual Awareness',
    'Rater Confidence',
    'Model Confidence',
    'Communication & Clarity'
]


def load_data():
    """Load and preprocess the evaluation data."""
    df = pd.read_excel(DATA_PATH)

    # Clean rater names (fix encoding issues)
    name_mapping = {
        'RaphaÃ«l Burger': 'Raphaël Burger',
        'AurÃ©lie Leuenberger': 'Aurélie Leuenberger'
    }
    df['Name'] = df['Name'].replace(name_mapping)

    return df


def create_table1(df):
    """Create Table 1: Descriptive statistics of the evaluation dataset."""

    table1_data = []

    # Overall statistics
    table1_data.append({
        'Category': 'Overall',
        'Metric': 'Total ratings',
        'Value': len(df)
    })
    table1_data.append({
        'Category': 'Overall',
        'Metric': 'Unique questions',
        'Value': df['Question'].nunique()
    })
    table1_data.append({
        'Category': 'Overall',
        'Metric': 'Expert raters',
        'Value': df['Name'].nunique()
    })

    # Ratings per question
    ratings_per_q = df.groupby('Question').size()
    table1_data.append({
        'Category': 'Ratings per question',
        'Metric': 'Mean (SD)',
        'Value': f"{ratings_per_q.mean():.2f} ({ratings_per_q.std():.2f})"
    })
    table1_data.append({
        'Category': 'Ratings per question',
        'Metric': 'Median [IQR]',
        'Value': f"{ratings_per_q.median():.0f} [{ratings_per_q.quantile(0.25):.0f}-{ratings_per_q.quantile(0.75):.0f}]"
    })
    table1_data.append({
        'Category': 'Ratings per question',
        'Metric': 'Range',
        'Value': f"{ratings_per_q.min()}-{ratings_per_q.max()}"
    })

    # Pre-alignment (First) ratings
    table1_data.append({
        'Category': 'Pre-alignment evaluation',
        'Metric': 'Total ratings',
        'Value': df[EVAL_COLS[0]].notna().sum()
    })

    # Post-alignment (Second) ratings
    has_second = df[SECOND_EVAL_COLS[0]].notna().sum()
    questions_with_second = df[df[SECOND_EVAL_COLS[0]].notna()]['Question'].nunique()
    table1_data.append({
        'Category': 'Post-alignment evaluation',
        'Metric': 'Total ratings',
        'Value': has_second
    })
    table1_data.append({
        'Category': 'Post-alignment evaluation',
        'Metric': 'Questions with paired ratings',
        'Value': questions_with_second
    })

    # Rater contributions
    rater_counts = df.groupby('Name').size().sort_values(ascending=False)
    for rater, count in rater_counts.items():
        second_count = df[df['Name'] == rater][SECOND_EVAL_COLS[0]].notna().sum()
        table1_data.append({
            'Category': 'Rater contributions',
            'Metric': rater,
            'Value': f"{count} ({second_count} paired)"
        })

    table1_df = pd.DataFrame(table1_data)
    return table1_df


def calculate_metrics(df):
    """Calculate main evaluation metrics for pre and post alignment."""

    results = []

    # Pre-alignment metrics
    for col, name in zip(EVAL_COLS, DIMENSION_NAMES):
        values = df[col].dropna()
        results.append({
            'Dimension': name,
            'Condition': 'Pre-alignment',
            'N': len(values),
            'Mean': values.mean(),
            'SD': values.std(),
            'Median': values.median(),
            'IQR_25': values.quantile(0.25),
            'IQR_75': values.quantile(0.75),
            'Min': values.min(),
            'Max': values.max()
        })

    # Post-alignment metrics
    for col, name in zip(SECOND_EVAL_COLS, DIMENSION_NAMES):
        values = df[col].dropna()
        if len(values) > 0:
            results.append({
                'Dimension': name,
                'Condition': 'Post-alignment',
                'N': len(values),
                'Mean': values.mean(),
                'SD': values.std(),
                'Median': values.median(),
                'IQR_25': values.quantile(0.25),
                'IQR_75': values.quantile(0.75),
                'Min': values.min(),
                'Max': values.max()
            })

    return pd.DataFrame(results)


def calculate_paired_comparison(df):
    """Calculate paired comparison between pre and post alignment ratings."""

    # Filter to rows with both pre and post ratings
    paired_mask = df[SECOND_EVAL_COLS[0]].notna()
    paired_df = df[paired_mask].copy()

    results = []

    for pre_col, post_col, name in zip(EVAL_COLS, SECOND_EVAL_COLS, DIMENSION_NAMES):
        pre_values = paired_df[pre_col].values
        post_values = paired_df[post_col].values

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(pre_values, post_values)

        # Wilcoxon signed-rank test (non-parametric)
        try:
            w_stat, w_p_value = stats.wilcoxon(pre_values, post_values)
        except:
            w_stat, w_p_value = np.nan, np.nan

        # Effect size (Cohen's d for paired samples)
        diff = post_values - pre_values
        cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0

        results.append({
            'Dimension': name,
            'Pre-alignment Mean (SD)': f"{pre_values.mean():.2f} ({pre_values.std():.2f})",
            'Post-alignment Mean (SD)': f"{post_values.mean():.2f} ({post_values.std():.2f})",
            'Difference': f"{diff.mean():.2f}",
            'Cohen\'s d': f"{cohens_d:.2f}",
            't-statistic': f"{t_stat:.2f}",
            'p-value': f"{p_value:.4f}" if p_value >= 0.0001 else "<0.0001",
            'Wilcoxon p': f"{w_p_value:.4f}" if w_p_value >= 0.0001 else "<0.0001"
        })

    return pd.DataFrame(results)


def calculate_inter_rater_reliability(df):
    """Calculate inter-rater reliability metrics."""

    from itertools import combinations

    results = []

    for col, name in zip(EVAL_COLS, DIMENSION_NAMES):
        # Get ratings by question
        question_ratings = df.groupby('Question')[col].apply(list)

        # Filter questions with multiple raters
        multi_rater_q = question_ratings[question_ratings.apply(len) >= 2]

        if len(multi_rater_q) == 0:
            continue

        # Calculate pairwise agreement
        agreements = []
        for ratings in multi_rater_q:
            for r1, r2 in combinations(ratings, 2):
                agreements.append(abs(r1 - r2))

        mean_abs_diff = np.mean(agreements)
        exact_agreement = sum(1 for a in agreements if a == 0) / len(agreements)
        within_1_agreement = sum(1 for a in agreements if a <= 1) / len(agreements)

        results.append({
            'Dimension': name,
            'Questions with ≥2 raters': len(multi_rater_q),
            'Pairwise comparisons': len(agreements),
            'Mean absolute difference': f"{mean_abs_diff:.2f}",
            'Exact agreement (%)': f"{exact_agreement*100:.1f}",
            'Within 1-point agreement (%)': f"{within_1_agreement*100:.1f}"
        })

    return pd.DataFrame(results)


def plot_dimension_distributions(df):
    """Create violin plots for each evaluation dimension."""

    fig, axes = plt.subplots(2, 5, figsize=(16, 8))
    axes = axes.flatten()

    for idx, (col, name) in enumerate(zip(EVAL_COLS, DIMENSION_NAMES)):
        ax = axes[idx]

        # Pre-alignment
        pre_values = df[col].dropna()

        # Post-alignment
        post_col = SECOND_EVAL_COLS[idx]
        post_values = df[post_col].dropna()

        # Create data for violin plot
        data_list = [pre_values]
        labels = ['Pre-alignment']

        if len(post_values) > 0:
            data_list.append(post_values)
            labels.append('Post-alignment')

        parts = ax.violinplot(data_list, positions=range(len(data_list)),
                             showmeans=True, showmedians=True)

        # Color the violins
        colors = ['#3498db', '#e74c3c']
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Score (1-5)')
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.set_ylim(0.5, 5.5)
        ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle('Evaluation Dimension Distributions: Pre vs Post Alignment',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dimension_distributions.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'dimension_distributions.pdf', bbox_inches='tight')
    plt.close()


def plot_paired_comparison(df):
    """Create bar plot comparing pre and post alignment scores."""

    paired_mask = df[SECOND_EVAL_COLS[0]].notna()
    paired_df = df[paired_mask].copy()

    pre_means = []
    post_means = []
    pre_sems = []
    post_sems = []

    for pre_col, post_col in zip(EVAL_COLS, SECOND_EVAL_COLS):
        pre_vals = paired_df[pre_col]
        post_vals = paired_df[post_col]

        pre_means.append(pre_vals.mean())
        post_means.append(post_vals.mean())
        pre_sems.append(pre_vals.sem())
        post_sems.append(post_vals.sem())

    x = np.arange(len(DIMENSION_NAMES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(x - width/2, pre_means, width, yerr=pre_sems,
                   label='Pre-alignment', color='#3498db', capsize=3, alpha=0.8)
    bars2 = ax.bar(x + width/2, post_means, width, yerr=post_sems,
                   label='Post-alignment', color='#e74c3c', capsize=3, alpha=0.8)

    ax.set_ylabel('Mean Score (1-5)', fontsize=12)
    ax.set_xlabel('Evaluation Dimension', fontsize=12)
    ax.set_title('Pre vs Post Alignment Evaluation Scores (Paired Ratings Only, n=130)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(DIMENSION_NAMES, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 5.5)
    ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5, label='Neutral')

    # Add significance markers
    for i, (pre, post) in enumerate(zip(EVAL_COLS, SECOND_EVAL_COLS)):
        pre_vals = paired_df[pre]
        post_vals = paired_df[post]
        _, p_val = stats.ttest_rel(pre_vals, post_vals)

        max_height = max(pre_means[i] + pre_sems[i], post_means[i] + post_sems[i])

        if p_val < 0.001:
            sig = '***'
        elif p_val < 0.01:
            sig = '**'
        elif p_val < 0.05:
            sig = '*'
        else:
            sig = 'ns'

        ax.text(i, max_height + 0.15, sig, ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'paired_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'paired_comparison.pdf', bbox_inches='tight')
    plt.close()


def plot_rater_heatmap(df):
    """Create heatmap of mean scores by rater and dimension."""

    rater_means = df.groupby('Name')[EVAL_COLS].mean()
    rater_means.columns = DIMENSION_NAMES

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(rater_means, annot=True, fmt='.2f', cmap='RdYlGn',
                center=3, vmin=1, vmax=5, ax=ax,
                cbar_kws={'label': 'Mean Score'})

    ax.set_title('Mean Evaluation Scores by Rater and Dimension (Pre-alignment)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Evaluation Dimension', fontsize=12)
    ax.set_ylabel('Rater', fontsize=12)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'rater_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'rater_heatmap.pdf', bbox_inches='tight')
    plt.close()


def plot_score_distributions(df):
    """Create histogram of overall score distribution."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pre-alignment overall mean per rating
    pre_overall = df[EVAL_COLS].mean(axis=1)
    axes[0].hist(pre_overall, bins=20, edgecolor='black', alpha=0.7, color='#3498db')
    axes[0].axvline(pre_overall.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {pre_overall.mean():.2f}')
    axes[0].set_xlabel('Mean Score Across All Dimensions')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Pre-alignment Overall Score Distribution', fontweight='bold')
    axes[0].legend()

    # Post-alignment overall mean per rating (where available)
    post_mask = df[SECOND_EVAL_COLS[0]].notna()
    if post_mask.sum() > 0:
        post_overall = df.loc[post_mask, SECOND_EVAL_COLS].mean(axis=1)
        axes[1].hist(post_overall, bins=20, edgecolor='black', alpha=0.7, color='#e74c3c')
        axes[1].axvline(post_overall.mean(), color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {post_overall.mean():.2f}')
        axes[1].set_xlabel('Mean Score Across All Dimensions')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Post-alignment Overall Score Distribution', fontweight='bold')
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'score_distributions.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'score_distributions.pdf', bbox_inches='tight')
    plt.close()


def plot_dimension_correlation(df):
    """Create correlation heatmap between dimensions."""

    corr_matrix = df[EVAL_COLS].corr()
    corr_matrix.index = DIMENSION_NAMES
    corr_matrix.columns = DIMENSION_NAMES

    fig, ax = plt.subplots(figsize=(10, 8))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1, ax=ax, square=True,
                cbar_kws={'label': 'Correlation'})

    ax.set_title('Correlation Between Evaluation Dimensions (Pre-alignment)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dimension_correlation.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'dimension_correlation.pdf', bbox_inches='tight')
    plt.close()


def generate_summary_stats(df):
    """Generate overall summary statistics."""

    summary = {}

    # Pre-alignment overall
    pre_all = df[EVAL_COLS].values.flatten()
    pre_all = pre_all[~np.isnan(pre_all)]
    summary['pre_alignment'] = {
        'n_ratings': len(df),
        'n_questions': df['Question'].nunique(),
        'overall_mean': pre_all.mean(),
        'overall_std': pre_all.std(),
        'overall_median': np.median(pre_all)
    }

    # Post-alignment overall
    post_mask = df[SECOND_EVAL_COLS[0]].notna()
    if post_mask.sum() > 0:
        post_all = df.loc[post_mask, SECOND_EVAL_COLS].values.flatten()
        post_all = post_all[~np.isnan(post_all)]
        summary['post_alignment'] = {
            'n_ratings': post_mask.sum(),
            'n_questions': df.loc[post_mask, 'Question'].nunique(),
            'overall_mean': post_all.mean(),
            'overall_std': post_all.std(),
            'overall_median': np.median(post_all)
        }

    return summary


def main():
    """Main analysis pipeline."""

    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} ratings for {df['Question'].nunique()} questions")

    # Generate Table 1
    print("\nGenerating Table 1...")
    table1 = create_table1(df)
    print(table1.to_string(index=False))
    table1.to_csv(OUTPUT_DIR / 'table1_descriptive_stats.csv', index=False)

    # Calculate metrics
    print("\nCalculating evaluation metrics...")
    metrics_df = calculate_metrics(df)
    metrics_df.to_csv(OUTPUT_DIR / 'evaluation_metrics.csv', index=False)

    # Create formatted metrics table
    print("\n=== Evaluation Metrics Summary ===")
    for condition in ['Pre-alignment', 'Post-alignment']:
        cond_df = metrics_df[metrics_df['Condition'] == condition]
        if len(cond_df) > 0:
            print(f"\n{condition}:")
            display_df = cond_df[['Dimension', 'N', 'Mean', 'SD', 'Median']].copy()
            display_df['Mean'] = display_df['Mean'].round(2)
            display_df['SD'] = display_df['SD'].round(2)
            print(display_df.to_string(index=False))

    # Paired comparison
    print("\nCalculating paired comparison (pre vs post alignment)...")
    paired_df = calculate_paired_comparison(df)
    print("\n=== Paired Comparison (n=130) ===")
    print(paired_df.to_string(index=False))
    paired_df.to_csv(OUTPUT_DIR / 'paired_comparison.csv', index=False)

    # Inter-rater reliability
    print("\nCalculating inter-rater reliability...")
    irr_df = calculate_inter_rater_reliability(df)
    print("\n=== Inter-Rater Reliability ===")
    print(irr_df.to_string(index=False))
    irr_df.to_csv(OUTPUT_DIR / 'inter_rater_reliability.csv', index=False)

    # Summary statistics
    summary = generate_summary_stats(df)
    print("\n=== Overall Summary ===")
    print(f"Pre-alignment: Mean = {summary['pre_alignment']['overall_mean']:.2f} "
          f"(SD = {summary['pre_alignment']['overall_std']:.2f})")
    if 'post_alignment' in summary:
        print(f"Post-alignment: Mean = {summary['post_alignment']['overall_mean']:.2f} "
              f"(SD = {summary['post_alignment']['overall_std']:.2f})")

    # Generate plots
    print("\nGenerating plots...")
    plot_dimension_distributions(df)
    print("  - Dimension distributions")

    plot_paired_comparison(df)
    print("  - Paired comparison")

    plot_rater_heatmap(df)
    print("  - Rater heatmap")

    plot_score_distributions(df)
    print("  - Score distributions")

    plot_dimension_correlation(df)
    print("  - Dimension correlation")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")

    return df, table1, metrics_df, paired_df, irr_df, summary


if __name__ == '__main__':
    df, table1, metrics_df, paired_df, irr_df, summary = main()
