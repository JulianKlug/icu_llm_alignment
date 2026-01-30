"""Data loading and preprocessing utilities."""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path('/mnt/data1/klug/datasets/meditron/Meditron_ICU.xlsx')

# Evaluation dimension column names
FIRST_EVAL_COLS = [
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

# Short dimension names for plotting
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


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load and clean the Meditron ICU dataset.

    Returns:
        DataFrame with cleaned data
    """
    df = pd.read_excel(path)

    # Fix encoding issues in rater names
    df['Name'] = df['Name'].str.replace('Ã«', 'ë').str.replace('Ã©', 'é')

    # Create a unique question identifier
    df['question_id'] = df.groupby('Question').ngroup()

    return df


def get_eval_columns(answer: str = 'first') -> list:
    """
    Get evaluation column names for first or second answer.

    Args:
        answer: 'first' or 'second'

    Returns:
        List of column names
    """
    if answer.lower() == 'first':
        return FIRST_EVAL_COLS
    elif answer.lower() == 'second':
        return SECOND_EVAL_COLS
    else:
        raise ValueError("answer must be 'first' or 'second'")


def reshape_for_agreement(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Reshape data for interrater agreement analysis.

    Creates a matrix where rows=questions, columns=raters, values=ratings.

    Args:
        df: Input DataFrame
        column: Column to use for values (e.g., 'Vote' or an eval column)

    Returns:
        Pivoted DataFrame (questions x raters)
    """
    # Pivot table: questions as rows, raters as columns
    pivot = df.pivot_table(
        index='question_id',
        columns='Name',
        values=column,
        aggfunc='first'  # In case of duplicates, take first
    )

    return pivot


def get_rated_answers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get evaluation scores for the rated answer(s) based on Vote.

    - Vote=1: Use first answer evals
    - Vote=2: Use second answer evals
    - Vote=12: Use both (create two rows)

    Returns:
        DataFrame with standardized eval columns
    """
    rows = []

    for idx, row in df.iterrows():
        vote = row['Vote']
        base_data = {
            'question_id': row['question_id'],
            'Question': row['Question'],
            'Name': row['Name'],
            'Vote': vote,
        }

        if vote == 1 or vote == 12:
            # First answer rated
            first_row = base_data.copy()
            first_row['answer_type'] = 'first'
            for i, col in enumerate(FIRST_EVAL_COLS):
                first_row[DIMENSION_NAMES[i]] = row[col]
            rows.append(first_row)

        if vote == 2 or vote == 12:
            # Second answer rated
            second_row = base_data.copy()
            second_row['answer_type'] = 'second'
            for i, col in enumerate(SECOND_EVAL_COLS):
                second_row[DIMENSION_NAMES[i]] = row[col]
            rows.append(second_row)

    return pd.DataFrame(rows)


def get_summary_stats(series: pd.Series) -> dict:
    """
    Calculate summary statistics for a series.

    Returns:
        Dictionary with mean, std, median, q25, q75, min, max, n
    """
    return {
        'n': series.count(),
        'mean': series.mean(),
        'std': series.std(),
        'median': series.median(),
        'q25': series.quantile(0.25),
        'q75': series.quantile(0.75),
        'min': series.min(),
        'max': series.max(),
    }


# Standardized eval column names (for concatenated answers)
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

    This combines all answers (first and second) into a single dataset where each
    row represents one rater's evaluation of one answer.

    Returns:
        DataFrame with columns: Answer, Question, Name, answer_type, and standardized Eval columns
    """
    # First answer dataframe
    first_answer_df = df[['First Answer', 'Question', 'Name', 'question_id'] + FIRST_EVAL_COLS].copy()
    first_answer_df = first_answer_df.rename(columns={'First Answer': 'Answer'})
    first_answer_df['answer_type'] = 'first'

    # Second answer dataframe - rename columns to match first answer
    second_cols_map = dict(zip(SECOND_EVAL_COLS, EVAL_COLS))
    second_cols_map['Second Answer'] = 'Answer'

    second_answer_df = df[['Second Answer', 'Question', 'Name', 'question_id'] + SECOND_EVAL_COLS].copy()
    second_answer_df = second_answer_df.rename(columns=second_cols_map)
    second_answer_df['answer_type'] = 'second'

    # Concatenate
    all_answers_df = pd.concat([first_answer_df, second_answer_df], ignore_index=True)

    # Remove rows where Answer is NaN
    all_answers_df = all_answers_df.dropna(subset=['Answer'])

    return all_answers_df
