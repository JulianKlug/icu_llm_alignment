#!/usr/bin/env python3
"""
00_data_dictionary.py
=====================
Data exploration and column documentation for Meditron ICU evaluation dataset.

This script:
1. Loads and explores the dataset
2. Creates a data dictionary documenting all columns
3. Validates data integrity
4. Outputs summary statistics

Output:
- output/data_dictionary.md
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from analyses.utils import load_data, DIMENSION_NAMES, FIRST_EVAL_COLS, SECOND_EVAL_COLS

OUTPUT_DIR = project_root / 'output'


def create_data_dictionary(df: pd.DataFrame) -> str:
    """Create markdown documentation for the dataset."""

    lines = [
        "# Meditron ICU Dataset - Data Dictionary",
        "",
        "## Overview",
        "",
        f"- **Total rows**: {len(df)}",
        f"- **Total columns**: {len(df.columns)}",
        f"- **Unique questions**: {df['Question'].nunique()}",
        f"- **Unique raters**: {df['Name'].nunique()}",
        f"- **Date range**: {df['Created At'].min()} to {df['Created At'].max()}",
        "",
        "## Rater Distribution",
        "",
        "| Rater | Ratings |",
        "|-------|---------|",
    ]

    for name, count in df['Name'].value_counts().items():
        lines.append(f"| {name} | {count} |")

    lines.extend([
        "",
        "## Vote Distribution",
        "",
        "| Vote | Meaning | Count | Percentage |",
        "|------|---------|-------|------------|",
    ])

    vote_counts = df['Vote'].value_counts().sort_index()
    vote_meanings = {1: "First answer preferred", 2: "Second answer preferred", 12: "Both answers equal"}
    for vote, count in vote_counts.items():
        pct = count / len(df) * 100
        meaning = vote_meanings.get(vote, "Unknown")
        lines.append(f"| {vote} | {meaning} | {count} | {pct:.1f}% |")

    lines.extend([
        "",
        "## Column Descriptions",
        "",
        "### Metadata Columns",
        "",
        "| Column | Type | Description | Non-null |",
        "|--------|------|-------------|----------|",
    ])

    metadata_cols = {
        'Created At': 'Timestamp when the rating was created',
        'Question': 'The ICU clinical question text',
        'Specialty': 'Medical specialty (all ICU-related)',
        'Contribution ID': 'Internal contribution identifier',
        'Vote': 'Preferred answer: 1=First, 2=Second, 12=Both equal',
        'User ID': 'Internal user identifier',
        'Name': 'Rater name',
        'Is Review': 'Whether this is a review rating',
        'Review Count': 'Number of reviews',
        'Country': 'Country code',
        'Number of Tags': 'Number of tags assigned',
        'Tags': 'Binary-encoded tag values',
        'Working Group': 'Working group name',
        'question_id': 'Computed unique question identifier',
    }

    for col, desc in metadata_cols.items():
        if col in df.columns:
            dtype = str(df[col].dtype)
            non_null = df[col].notna().sum()
            lines.append(f"| {col} | {dtype} | {desc} | {non_null} |")

    lines.extend([
        "",
        "### Answer Columns",
        "",
        "| Column | Description |",
        "|--------|-------------|",
        "| First Answer | LLM's first generated answer |",
        "| Second Answer | LLM's second generated answer |",
        "| First Answer Improved | Expert's improved version of first answer (optional) |",
        "| Second Answer Improved | Expert's improved version of second answer (optional) |",
        "| Ideal Answer | Reference ideal answer (if available) |",
        "",
        "### Evaluation Dimensions (First Answer)",
        "",
        "All scores are on a 1-5 Likert scale.",
        "",
        "| Column | Dimension | Mean | Std | Min | Max |",
        "|--------|-----------|------|-----|-----|-----|",
    ])

    for col, name in zip(FIRST_EVAL_COLS, DIMENSION_NAMES):
        mean = df[col].mean()
        std = df[col].std()
        min_val = df[col].min()
        max_val = df[col].max()
        lines.append(f"| {col} | {name} | {mean:.2f} | {std:.2f} | {min_val} | {max_val} |")

    lines.extend([
        "",
        "### Evaluation Dimensions (Second Answer)",
        "",
        "| Column | Dimension | Mean | Std | Min | Max | Non-null |",
        "|--------|-----------|------|-----|-----|-----|----------|",
    ])

    for col, name in zip(SECOND_EVAL_COLS, DIMENSION_NAMES):
        mean = df[col].mean()
        std = df[col].std()
        min_val = df[col].min()
        max_val = df[col].max()
        non_null = df[col].notna().sum()
        lines.append(f"| {col} | {name} | {mean:.2f} | {std:.2f} | {min_val} | {max_val} | {non_null} |")

    lines.extend([
        "",
        "## Data Quality Notes",
        "",
    ])

    # Check for issues
    issues = []

    # Missing values in eval columns
    for col in FIRST_EVAL_COLS:
        missing = df[col].isna().sum()
        if missing > 0:
            issues.append(f"- {col}: {missing} missing values")

    # Out of range values
    for col in FIRST_EVAL_COLS:
        out_of_range = ((df[col] < 1) | (df[col] > 5)).sum()
        if out_of_range > 0:
            issues.append(f"- {col}: {out_of_range} values outside 1-5 range")

    if issues:
        lines.extend(issues)
    else:
        lines.append("- No data quality issues detected in first answer evaluations")

    # Second answer analysis
    lines.extend([
        "",
        "### Second Answer Evaluation Coverage",
        "",
    ])

    # Second evals should be present when Vote=2 or Vote=12
    expected_second = df[df['Vote'].isin([2, 12])].shape[0]
    actual_second = df[SECOND_EVAL_COLS[0]].notna().sum()
    lines.append(f"- Expected second answer evaluations (Vote=2 or 12): {expected_second}")
    lines.append(f"- Actual second answer evaluations present: {actual_second}")

    return "\n".join(lines)


def validate_data(df: pd.DataFrame) -> dict:
    """Validate data integrity and return summary."""

    validation = {
        'total_rows': len(df),
        'unique_questions': df['Question'].nunique(),
        'unique_raters': df['Name'].nunique(),
        'vote_distribution': df['Vote'].value_counts().to_dict(),
        'missing_first_evals': {col: df[col].isna().sum() for col in FIRST_EVAL_COLS},
        'missing_second_evals': {col: df[col].isna().sum() for col in SECOND_EVAL_COLS},
        'ratings_per_question': df.groupby('Question').size().describe().to_dict(),
    }

    return validation


def main():
    """Main function to create data dictionary."""

    print("=" * 60)
    print("00_data_dictionary.py - Creating Data Dictionary")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    df = load_data()
    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")

    # Validate data
    print("\n2. Validating data...")
    validation = validate_data(df)
    print(f"   - {validation['unique_questions']} unique questions")
    print(f"   - {validation['unique_raters']} unique raters")
    print(f"   - Vote distribution: {validation['vote_distribution']}")

    # Create data dictionary
    print("\n3. Creating data dictionary...")
    dictionary_md = create_data_dictionary(df)

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / 'data_dictionary.md'
    with open(output_path, 'w') as f:
        f.write(dictionary_md)
    print(f"   Saved to: {output_path}")

    # Print summary statistics
    print("\n4. Summary Statistics:")
    print(f"   - Mean ratings per question: {validation['ratings_per_question']['mean']:.1f}")
    print(f"   - Min ratings per question: {validation['ratings_per_question']['min']:.0f}")
    print(f"   - Max ratings per question: {validation['ratings_per_question']['max']:.0f}")

    print("\n" + "=" * 60)
    print("Data dictionary created successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
