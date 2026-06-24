from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pandas as pd
from fastmcp import FastMCP

from edsmith.data.parser import COMPONENT_HEADINGS
from edsmith.data.loader import apply_size_limit, train_test_split
from edsmith.examiner.feedback import generate_feedback
from edsmith.providers.openrouter import OpenRouterProvider
from edsmith.session.state import load_state

_DEFAULT_DRIVE = "/content/drive/MyDrive/edsmith"


def _drive_path() -> Path:
    return Path(os.environ.get("EDSMITH_DRIVE_PATH", _DEFAULT_DRIVE))


def _ensure_session_data(drive_path: Path, session_id: str, state) -> pd.DataFrame:
    """Return the training DataFrame, initialising session data on first call."""
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
        raw_test = raw_test.sample(frac=s.test_ratio, random_state=s.random_state).reset_index(drop=True)

    train_df, val_df = train_test_split(raw_train, validation_ratio=s.validation_ratio, random_state=s.random_state)

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


def register_examiner_pass(app: FastMCP):
    @app.tool(
        title="Run Examiner Pass",
        description=(
            "Generate per-component Feedback for all training Essays in one iteration. "
            "Reads SessionState from disk, initialises session data on first call, "
            "runs generate_feedback concurrently across all Essays, and writes a "
            "feedback parquet to the session directory. Returns an ExaminerSummary."
        ),
    )
    async def run_examiner_pass(
        session_id: str,
        iteration: int,
        concurrency: int = 4,
    ) -> dict:
        drive_path = _drive_path()
        state = load_state(drive_path, session_id)
        train_df = await asyncio.to_thread(_ensure_session_data, drive_path, session_id, state)

        provider = OpenRouterProvider()
        semaphore = asyncio.Semaphore(concurrency)
        records: list[dict] = []
        warnings: list[str] = []

        async def process_essay(row: dict) -> None:
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

        await asyncio.gather(*[
            process_essay(row) for row in train_df.to_dict(orient="records")
        ])

        feedback_df = pd.DataFrame(records)
        out_path = drive_path / "sessions" / session_id / f"feedback_iter{iteration}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        feedback_df.to_parquet(out_path, index=False)

        return _build_summary(feedback_df, len(train_df), warnings, out_path, session_id, iteration)

    return run_examiner_pass
