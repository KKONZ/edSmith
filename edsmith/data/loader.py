from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from datasets import config as ds_config
from datasets import load_dataset

from edsmith.data.parser import ParsedEvaluation, parse_evaluation

_HF_DATASET = "chillies/IELTS-writing-task-2-evaluation"

_COLUMN_MAP = {
    "prompt": "question",
}


def clear_dataset_cache() -> None:
    """Delete the local HuggingFace cache for the IELTS dataset."""
    cache_dir = Path(ds_config.HF_DATASETS_CACHE) / "chillies___ielts-writing-task-2-evaluation"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)


def load_ielts(split: str = "train", force_download: bool = False) -> pd.DataFrame:
    """Load the IELTS dataset and rename columns to match domain vocabulary.

    Raw columns:  prompt, essay, evaluation, band
    Domain cols:  question, essay, evaluation, band
    """
    if force_download:
        clear_dataset_cache()
    ds = load_dataset(_HF_DATASET, split=split)
    df = ds.to_pandas().rename(columns=_COLUMN_MAP)
    return df


def load_with_parsed_evaluations(split: str = "train", force_download: bool = False) -> pd.DataFrame:
    """Load and attach a ParsedEvaluation object for every row."""
    df = load_ielts(split=split, force_download=force_download)
    df["parsed_evaluation"] = df["evaluation"].apply(parse_evaluation)
    return df


def build_baseline_feedback_df(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten parsed evaluations into the same format Phase 1 produces.

    Uses the original human-written component text and scores as training
    targets for the baseline Scorer (Session 0).
    Rows with a null component score are dropped — they cannot be labelled.
    """
    records = []
    for _, row in df.iterrows():
        parsed = row["parsed_evaluation"]
        for component, eval_ in parsed.components.items():
            if eval_.score is None:
                continue
            records.append({
                "question": row["question"],
                "essay": row["essay"],
                "band": row["band"],
                "component": component,
                "feedback_text": eval_.text,
                "score": eval_.score,
            })
    return pd.DataFrame(records)


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
