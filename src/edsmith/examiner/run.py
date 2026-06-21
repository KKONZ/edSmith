"""Batch examiner pass — generates Feedback for all training essays in one iteration."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pandas as pd

from edsmith.data.loader import apply_size_limit, train_test_split
from edsmith.data.parser import COMPONENT_HEADINGS
from edsmith.examiner.feedback import generate_feedback
from edsmith.providers.base import LLMProvider
from edsmith.session.state import load_state


def _ensure_session_data(drive_path: Path, session_id: str, state) -> pd.DataFrame:
    data_dir = drive_path / "sessions" / session_id / "data"
    train_path = data_dir / "train.parquet"

    if train_path.exists():
        return pd.read_parquet(train_path)

    from edsmith.data.loader import load_ielts

    data_dir.mkdir(parents=True, exist_ok=True)
    s = state.sampling
    raw_train = load_ielts("train")
    raw_test = load_ielts("test")

    if s.size:
        raw_train, raw_test = apply_size_limit(raw_train, raw_test, s.size, s.random_state)
    if s.test_ratio is not None:
        raw_test = raw_test.sample(
            frac=s.test_ratio, random_state=s.random_state
        ).reset_index(drop=True)

    train_df, val_df = train_test_split(
        raw_train, validation_ratio=s.validation_ratio, random_state=s.random_state
    )

    train_df.to_parquet(data_dir / "train.parquet", index=False)
    val_df.to_parquet(data_dir / "val.parquet", index=False)
    raw_test.to_parquet(data_dir / "test.parquet", index=False)

    return train_df


def _build_summary(
    feedback_df: pd.DataFrame,
    n_essays: int,
    warnings: list[str],
    parquet_path: Path,
    session_id: str,
    iteration: int,
) -> dict:
    essays_with_all = (
        feedback_df.groupby("essay")["component"]
        .nunique()
        .eq(len(COMPONENT_HEADINGS))
        .sum()
        if not feedback_df.empty
        else 0
    )
    score_distributions: dict[str, dict] = {}
    if not feedback_df.empty:
        for component in COMPONENT_HEADINGS:
            scores = feedback_df.loc[
                feedback_df["component"] == component, "score"
            ].dropna()
            if not scores.empty:
                score_distributions[component] = {
                    "mean": round(float(scores.mean()), 3),
                    "std": round(float(scores.std()), 3),
                    "count": int(scores.count()),
                }
    return {
        "session_id": session_id,
        "iteration": iteration,
        "essays_processed": int(feedback_df["essay"].nunique()) if not feedback_df.empty else 0,
        "essays_total": n_essays,
        "components_covered": int(essays_with_all),
        "score_distributions": score_distributions,
        "warnings": warnings,
        "parquet_path": str(parquet_path),
    }


async def run_examiner_pass(
    session_id: str,
    iteration: int,
    drive_path: Path,
    concurrency: int = 4,
    provider: LLMProvider | None = None,
) -> dict:
    """Generate per-component Feedback for all training essays in one iteration.

    Reads SessionState from disk, initialises session data splits on first call,
    runs generate_feedback concurrently across all essays, and writes a feedback
    parquet to the session directory. Returns an ExaminerSummary dict.

    provider — inject an LLMProvider for testing; defaults to OpenRouterProvider.
    """
    state = load_state(drive_path, session_id)
    train_df = await asyncio.to_thread(_ensure_session_data, drive_path, session_id, state)

    if provider is None:
        from edsmith.providers.openrouter import OpenRouterProvider
        provider = OpenRouterProvider()

    n_essays = len(train_df)
    semaphore = asyncio.Semaphore(concurrency)
    records: list[dict] = []
    warnings: list[str] = []
    completed = 0

    async def process_essay(row: dict) -> None:
        nonlocal completed
        async with semaphore:
            try:
                feedbacks = await generate_feedback(
                    question=row["question"],
                    essay=row["essay"],
                    policies=state.policies,
                    strategy=state.strategy_guidance,
                    provider=provider,
                    model_config=state.models,
                )
                for component, fb in feedbacks.items():
                    records.append({
                        "question": row["question"],
                        "essay": row["essay"],
                        "band": row.get("band"),
                        "component": component,
                        "feedback_text": fb.feedback,
                        "score": fb.score,
                        "tag": fb.tag,
                    })
            except Exception as exc:
                warnings.append(f"Essay failed: {exc}")
            finally:
                completed += 1
                if completed % 10 == 0 or completed == n_essays:
                    print(f"  [{completed}/{n_essays}]", end="\r", flush=True, file=sys.stderr)

    await asyncio.gather(*[process_essay(row) for row in train_df.to_dict(orient="records")])
    print(file=sys.stderr)  # newline after progress

    feedback_df = pd.DataFrame(records)
    out_path = drive_path / "sessions" / session_id / f"feedback_iter{iteration}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feedback_df.to_parquet(out_path, index=False)

    return _build_summary(feedback_df, n_essays, warnings, out_path, session_id, iteration)
