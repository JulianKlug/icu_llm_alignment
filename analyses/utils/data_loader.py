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

    The FIRST_EVAL_COLS always contain the evaluation of the PREFERRED answer.
    The SECOND_EVAL_COLS contain the evaluation of the other answer only when Vote=12.

    - Vote=1: FIRST_EVAL_COLS rate the first answer (preferred)
    - Vote=2: FIRST_EVAL_COLS rate the second answer (preferred)
    - Vote=12: FIRST_EVAL_COLS rate the first answer, SECOND_EVAL_COLS rate the second

    Returns:
        DataFrame with standardized eval columns and answer_type label
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

        if vote == 1:
            # First answer preferred: FIRST_EVAL_COLS rate the first answer
            r = base_data.copy()
            r['answer_type'] = 'first'
            for i, col in enumerate(FIRST_EVAL_COLS):
                r[DIMENSION_NAMES[i]] = row[col]
            rows.append(r)

        elif vote == 2:
            # Second answer preferred: FIRST_EVAL_COLS rate the second answer
            r = base_data.copy()
            r['answer_type'] = 'second'
            for i, col in enumerate(FIRST_EVAL_COLS):
                r[DIMENSION_NAMES[i]] = row[col]
            rows.append(r)

        elif vote == 12:
            # Both equal: FIRST_EVAL_COLS rate first, SECOND_EVAL_COLS rate second
            first_row = base_data.copy()
            first_row['answer_type'] = 'first'
            for i, col in enumerate(FIRST_EVAL_COLS):
                first_row[DIMENSION_NAMES[i]] = row[col]
            rows.append(first_row)

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
    Concatenate rated answers into one dataframe with standardized columns.

    The FIRST_EVAL_COLS always contain the evaluation of the PREFERRED answer.
    The SECOND_EVAL_COLS contain the evaluation of the other answer only when Vote=12.

    - Vote=1: FIRST_EVAL_COLS paired with First Answer text (first answer preferred)
    - Vote=2: FIRST_EVAL_COLS paired with Second Answer text (second answer preferred)
    - Vote=12: FIRST_EVAL_COLS paired with First Answer, SECOND_EVAL_COLS with Second Answer

    Returns:
        DataFrame with columns: Answer, Question, Name, answer_type, question_id,
        and standardized Eval columns
    """
    parts = []

    # Vote=1: first answer preferred, FIRST_EVAL_COLS rate the first answer
    vote1 = df[df['Vote'] == 1].copy()
    if len(vote1) > 0:
        v1_df = vote1[['First Answer', 'Question', 'Name', 'question_id'] + FIRST_EVAL_COLS].copy()
        v1_df = v1_df.rename(columns={'First Answer': 'Answer'})
        v1_df['answer_type'] = 'first'
        parts.append(v1_df)

    # Vote=2: second answer preferred, FIRST_EVAL_COLS rate the second answer
    vote2 = df[df['Vote'] == 2].copy()
    if len(vote2) > 0:
        v2_df = vote2[['Second Answer', 'Question', 'Name', 'question_id'] + FIRST_EVAL_COLS].copy()
        v2_df = v2_df.rename(columns={'Second Answer': 'Answer'})
        v2_df['answer_type'] = 'second'
        parts.append(v2_df)

    # Vote=12: both answers rated
    vote12 = df[df['Vote'] == 12].copy()
    if len(vote12) > 0:
        # First answer: FIRST_EVAL_COLS
        v12_first = vote12[['First Answer', 'Question', 'Name', 'question_id'] + FIRST_EVAL_COLS].copy()
        v12_first = v12_first.rename(columns={'First Answer': 'Answer'})
        v12_first['answer_type'] = 'first'
        parts.append(v12_first)

        # Second answer: SECOND_EVAL_COLS -> rename to standard EVAL_COLS
        second_cols_map = dict(zip(SECOND_EVAL_COLS, EVAL_COLS))
        second_cols_map['Second Answer'] = 'Answer'
        v12_second = vote12[['Second Answer', 'Question', 'Name', 'question_id'] + SECOND_EVAL_COLS].copy()
        v12_second = v12_second.rename(columns=second_cols_map)
        v12_second['answer_type'] = 'second'
        parts.append(v12_second)

    all_answers_df = pd.concat(parts, ignore_index=True)

    # Remove rows where Answer is NaN
    all_answers_df = all_answers_df.dropna(subset=['Answer'])

    return all_answers_df
