from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import pandas as pd
from fastmcp import FastMCP

from edsmith.data.parser import COMPONENT_HEADINGS
from edsmith.data.loader import apply_size_limit, train_test_split
from edsmith.examiner.feedback import generate_feedback
from edsmith.providers.openrouter import OpenRouterProvider
from edsmith.session.state import load_state

logger = logging.getLogger(__name__)

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

    if s.test_ratio is not None:
        raw_test = raw_test.sample(frac=s.test_ratio, random_state=s.random_state).reset_index(drop=True)
    if s.size:
        raw_train, raw_test = apply_size_limit(raw_train, raw_test, s.size, s.random_state)

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
    components: list[str],
) -> dict:
    essays_with_all = (
        feedback_df.groupby("essay")["component"]
        .nunique()
        .eq(len(components))
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
            "feedback parquet to the session directory. Returns an ExaminerSummary. "
            "Set retry_failed=True to skip essays already present in an existing parquet "
            "and append only the missing ones — useful for retrying a partial pass."
        ),
    )
    async def run_examiner_pass(
        session_id: str,
        iteration: int,
        concurrency: int = 12,
        retry_failed: bool = False,
    ) -> dict:
        import json as _json
        import sys

        drive_path = _drive_path()
        state = load_state(drive_path, session_id)
        train_df = await asyncio.to_thread(_ensure_session_data, drive_path, session_id, state)

        scorer_cfg_path = drive_path / "sessions" / session_id / "scorer_config.json"
        scorer_component: str | None = None
        if scorer_cfg_path.exists():
            scorer_component = _json.loads(scorer_cfg_path.read_text()).get("component")
        active_components = [scorer_component] if scorer_component else list(COMPONENT_HEADINGS.keys())

        out_path = drive_path / "sessions" / session_id / f"feedback_iter{iteration}.parquet"

        existing_df: pd.DataFrame | None = None
        if retry_failed and out_path.exists():
            existing_df = pd.read_parquet(out_path)
            already_done = set(existing_df["essay"].unique())
            train_df = train_df[~train_df["essay"].isin(already_done)].reset_index(drop=True)

        n_essays = len(train_df)
        expected_requests = n_essays * len(active_components)

        # File log for this pass — captures all edsmith.examiner.* loggers
        log_path = drive_path / "sessions" / session_id / f"examiner_iter{iteration}.log"
        _file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        _file_handler.setLevel(logging.INFO)
        _file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        _examiner_logger = logging.getLogger("edsmith.examiner")
        _examiner_logger.addHandler(_file_handler)
        _examiner_logger.setLevel(logging.INFO)

        logger.info(
            "START session=%s iter=%d essays=%d components=%s expected_requests=%d concurrency=%d",
            session_id, iteration, n_essays, active_components, expected_requests, concurrency,
        )
        print(f"[examiner] session={session_id} iter={iteration} essays={n_essays} components={active_components} expected_requests={expected_requests}", flush=True, file=sys.stderr)

        provider = OpenRouterProvider()
        semaphore = asyncio.Semaphore(concurrency)
        records: list[dict] = []
        warnings: list[str] = []
        completed = 0
        requests_done = 0

        async def process_essay(row: dict) -> None:
            nonlocal completed, requests_done
            async with semaphore:
                try:
                    try:
                        raw_band = row.get("band")
                        if raw_band is None:
                            band = None
                        elif str(raw_band).strip() == "<4":
                            band = 3.0
                        else:
                            band = float(raw_band)
                    except (TypeError, ValueError):
                        band = None
                    feedbacks = await generate_feedback(
                        question=row["question"],
                        essay=row["essay"],
                        policies=state.policies,
                        strategy=state.strategy_guidance,
                        provider=provider,
                        model_config=state.models,
                        band=band,
                        components=active_components,
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
                            "calibration_delta": fb.calibration_delta,
                        })
                        requests_done += 1
                except Exception as exc:
                    warnings.append(f"Essay failed: {exc}")
                finally:
                    completed += 1
                    logger.info(
                        "essay %d/%d done requests=%d/%d",
                        completed, n_essays, requests_done, expected_requests,
                    )
                    print(f"[examiner] {completed}/{n_essays} essays done", flush=True, file=sys.stderr)

        try:
            await asyncio.gather(*[
                process_essay(row) for row in train_df.to_dict(orient="records")
            ])
        finally:
            logger.info(
                "END essays=%d requests=%d/%d warnings=%d",
                completed, requests_done, expected_requests, len(warnings),
            )
            _examiner_logger.removeHandler(_file_handler)
            _file_handler.close()

        feedback_df = pd.DataFrame(records)
        if existing_df is not None:
            feedback_df = pd.concat([existing_df, feedback_df], ignore_index=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        feedback_df.to_parquet(out_path, index=False)

        return _build_summary(feedback_df, len(train_df), warnings, out_path, session_id, iteration, active_components)

    return run_examiner_pass


def register_calibrate_feedback(app: FastMCP):
    @app.tool(
        title="Calibrate Feedback",
        description=(
            "Run the score calibration pass on an existing feedback parquet. "
            "Loads feedback_iter{N}.parquet, groups records by essay, reconstructs "
            "ComponentFeedback objects, calls _reflect_and_calibrate for each essay "
            "whose scores deviate from the verified band, and writes the updated parquet "
            "in place. Returns a summary of how many essays were adjusted."
        ),
    )
    async def calibrate_feedback(
        session_id: str,
        iteration: int,
        concurrency: int = 12,
    ) -> dict:
        import sys
        from edsmith.examiner.feedback import ComponentFeedback, _reflect_and_calibrate

        drive_path = _drive_path()
        state = load_state(drive_path, session_id)
        out_path = drive_path / "sessions" / session_id / f"feedback_iter{iteration}.parquet"

        if not out_path.exists():
            raise FileNotFoundError(f"No feedback parquet found at {out_path}")

        df = pd.read_parquet(out_path)

        if "band" not in df.columns or df["band"].isna().all():
            raise ValueError("Parquet has no band column — cannot calibrate without verified bands.")

        provider = OpenRouterProvider()
        semaphore = asyncio.Semaphore(concurrency)

        essays_adjusted = 0
        essays_skipped_bad_band = 0
        essays_unchanged = 0
        updated_records: list[dict] = []

        async def calibrate_essay(essay_rows: list[dict]) -> list[dict]:
            nonlocal essays_adjusted, essays_skipped_bad_band, essays_unchanged
            band = essay_rows[0].get("band")

            feedbacks = {
                row["component"]: ComponentFeedback(
                    component=row["component"],
                    feedback=row.get("feedback_text") or "",
                    score=row.get("score"),
                    tag=row.get("tag"),
                    calibration_delta=row.get("calibration_delta", 0.0) or 0.0,
                )
                for row in essay_rows
            }

            try:
                if str(band).strip() == "<4":
                    band_float = 3.0
                else:
                    band_float = float(band)
            except (TypeError, ValueError):
                essays_skipped_bad_band += 1
                return essay_rows

            async with semaphore:
                calibrated = await _reflect_and_calibrate(
                    feedbacks=feedbacks,
                    band=band_float,
                    provider=provider,
                    model=state.models.generator,
                    enable_thinking=state.models.enable_thinking,
                )

            changed = any(
                calibrated[c].calibration_delta != feedbacks[c].calibration_delta
                for c in calibrated
            )
            if changed:
                essays_adjusted += 1
            else:
                essays_unchanged += 1

            return [
                {
                    **{k: row[k] for k in row if k not in ("feedback_text", "score", "tag", "calibration_delta")},
                    "feedback_text": calibrated[row["component"]].feedback,
                    "score": calibrated[row["component"]].score,
                    "tag": calibrated[row["component"]].tag,
                    "calibration_delta": calibrated[row["component"]].calibration_delta,
                }
                for row in essay_rows
                if row["component"] in calibrated
            ]

        groups: dict[str, list[dict]] = {}
        for row in df.to_dict(orient="records"):
            groups.setdefault(row["essay"], []).append(row)

        n_essays = len(groups)
        print(f"[calibrate] session={session_id} iter={iteration} essays={n_essays}", flush=True, file=sys.stderr)

        results = await asyncio.gather(*[calibrate_essay(rows) for rows in groups.values()])
        for essay_records in results:
            updated_records.extend(essay_records)

        updated_df = pd.DataFrame(updated_records)
        updated_df.to_parquet(out_path, index=False)

        return {
            "session_id": session_id,
            "iteration": iteration,
            "essays_total": n_essays,
            "essays_adjusted": essays_adjusted,
            "essays_unchanged": essays_unchanged,
            "essays_skipped_bad_band": essays_skipped_bad_band,
            "parquet_path": str(out_path),
        }

    return calibrate_feedback
