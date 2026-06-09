from __future__ import annotations

import pandas as pd
from datasets import load_dataset

from edsmith.data.parser import ParsedEvaluation, parse_evaluation

_HF_DATASET = "chillies/IELTS-writing-task-2-evaluation"

_COLUMN_MAP = {
    "prompt": "question",
}


def load_ielts(split: str = "train") -> pd.DataFrame:
    """Load the IELTS dataset and rename columns to match domain vocabulary.

    Raw columns:  prompt, essay, evaluation, band
    Domain cols:  question, essay, evaluation, band
    """
    ds = load_dataset(_HF_DATASET, split=split)
    df = ds.to_pandas().rename(columns=_COLUMN_MAP)
    return df


def load_with_parsed_evaluations(split: str = "train") -> pd.DataFrame:
    """Load and attach a ParsedEvaluation object for every row."""
    df = load_ielts(split=split)
    df["parsed_evaluation"] = df["evaluation"].apply(parse_evaluation)
    return df


def train_test_split(
    df: pd.DataFrame,
    validation_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a fixed validation set from the training DataFrame.

    The validation set is stratified by band score so all grade levels are
    represented.  The remaining rows form the working training set for the
    session.  Both splits are fixed for the duration of a session to eliminate
    random effects across iterations.
    """
    val = (
        df.groupby("band", group_keys=False)
        .apply(lambda g: g.sample(frac=validation_ratio, random_state=random_state))
    )
    train = df.drop(val.index)
    return train.reset_index(drop=True), val.reset_index(drop=True)
