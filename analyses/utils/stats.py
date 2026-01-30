"""Statistical functions for interrater agreement analysis."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def fleiss_kappa(ratings: np.ndarray) -> float:
    """
    Calculate Fleiss' Kappa for multiple raters.

    Args:
        ratings: 2D array where rows are subjects and columns are raters.
                 Values should be category labels (can have NaN for missing).

    Returns:
        Fleiss' Kappa coefficient
    """
    # Convert to DataFrame for easier handling
    if isinstance(ratings, pd.DataFrame):
        ratings = ratings.values

    # Get unique categories (excluding NaN)
    flat = ratings.flatten()
    categories = np.unique(flat[~pd.isna(flat)])
    n_categories = len(categories)

    n_subjects = ratings.shape[0]

    # Build the count matrix: for each subject, count how many raters chose each category
    count_matrix = np.zeros((n_subjects, n_categories))

    for i in range(n_subjects):
        row = ratings[i, :]
        valid = row[~pd.isna(row)]
        for j, cat in enumerate(categories):
            count_matrix[i, j] = np.sum(valid == cat)

    # Number of raters per subject
    n_raters_per_subject = count_matrix.sum(axis=1)

    # Filter out subjects with fewer than 2 raters
    valid_subjects = n_raters_per_subject >= 2
    count_matrix = count_matrix[valid_subjects]
    n_raters_per_subject = n_raters_per_subject[valid_subjects]
    n_subjects = count_matrix.shape[0]

    if n_subjects == 0:
        return np.nan

    # Mean number of raters
    n = n_raters_per_subject.mean()

    # P_j: proportion of all ratings that are category j
    total_ratings = count_matrix.sum()
    p_j = count_matrix.sum(axis=0) / total_ratings

    # P_i: extent of agreement for subject i
    # P_i = (1 / (n * (n-1))) * sum_j(n_ij * (n_ij - 1))
    P_i = np.zeros(n_subjects)
    for i in range(n_subjects):
        ni = n_raters_per_subject[i]
        if ni > 1:
            P_i[i] = (1 / (ni * (ni - 1))) * np.sum(count_matrix[i, :] * (count_matrix[i, :] - 1))
        else:
            P_i[i] = 1.0

    # P_bar: mean of P_i
    P_bar = P_i.mean()

    # P_e: expected agreement by chance
    P_e = np.sum(p_j ** 2)

    # Fleiss' Kappa
    if P_e == 1:
        return 1.0 if P_bar == 1 else 0.0

    kappa = (P_bar - P_e) / (1 - P_e)

    return kappa


def fleiss_kappa_per_question(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Calculate agreement score per question.

    For each question, calculates the proportion of raters who agreed on the
    majority vote (normalized measure of agreement).

    Args:
        df: DataFrame with columns 'question_id', 'Name', and the rating column
        column: Column name containing the ratings

    Returns:
        Series indexed by question_id with agreement scores (0-1)
    """
    agreement_scores = {}

    for qid, group in df.groupby('question_id'):
        ratings = group[column].dropna()
        if len(ratings) < 2:
            agreement_scores[qid] = np.nan
            continue

        # Count votes for each category
        counts = ratings.value_counts()
        max_count = counts.max()
        total = len(ratings)

        # Agreement = proportion who chose the most common option
        agreement_scores[qid] = max_count / total

    return pd.Series(agreement_scores)


def percent_agreement(ratings: np.ndarray) -> float:
    """
    Calculate overall percent agreement.

    Args:
        ratings: 2D array (subjects x raters)

    Returns:
        Percentage of subject pairs where raters agreed
    """
    if isinstance(ratings, pd.DataFrame):
        ratings = ratings.values

    n_subjects = ratings.shape[0]
    total_pairs = 0
    agree_pairs = 0

    for i in range(n_subjects):
        row = ratings[i, :]
        valid = row[~pd.isna(row)]
        n = len(valid)
        if n < 2:
            continue

        # Count pairs
        for j in range(n):
            for k in range(j + 1, n):
                total_pairs += 1
                if valid[j] == valid[k]:
                    agree_pairs += 1

    if total_pairs == 0:
        return np.nan

    return agree_pairs / total_pairs


def bootstrap_ci(
    data: np.ndarray,
    statistic_func: callable,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for a statistic.

    Args:
        data: Input data array
        statistic_func: Function that calculates the statistic
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level (e.g., 0.95 for 95% CI)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper)
    """
    np.random.seed(seed)

    n = len(data) if hasattr(data, '__len__') else data.shape[0]

    # Point estimate
    point_estimate = statistic_func(data)

    # Bootstrap samples
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        if isinstance(data, pd.DataFrame):
            sample_idx = np.random.choice(data.shape[0], size=data.shape[0], replace=True)
            sample = data.iloc[sample_idx]
        else:
            sample_idx = np.random.choice(n, size=n, replace=True)
            sample = data[sample_idx]

        try:
            stat = statistic_func(sample)
            if not np.isnan(stat):
                bootstrap_stats.append(stat)
        except:
            continue

    if len(bootstrap_stats) < 10:
        return point_estimate, np.nan, np.nan

    # Confidence interval
    alpha = 1 - ci
    ci_lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)

    return point_estimate, ci_lower, ci_upper


def interpret_kappa(kappa: float) -> str:
    """
    Interpret Kappa value according to Landis & Koch (1977).

    Args:
        kappa: Kappa coefficient

    Returns:
        Interpretation string
    """
    if kappa < 0:
        return "Poor (less than chance)"
    elif kappa < 0.20:
        return "Slight"
    elif kappa < 0.40:
        return "Fair"
    elif kappa < 0.60:
        return "Moderate"
    elif kappa < 0.80:
        return "Substantial"
    else:
        return "Almost perfect"
