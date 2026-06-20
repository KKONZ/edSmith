from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from fastmcp import FastMCP

from edsmith.chief_examiner.diagnostic import run_diagnostic
from edsmith.providers.openrouter import OpenRouterProvider
from edsmith.session.state import (
    load_metrics,
    load_proposal,
    load_state,
    save_proposal,
    save_state,
)

_DEFAULT_DRIVE = "/content/drive/MyDrive/edsmith"


def _drive_path() -> Path:
    return Path(os.environ.get("EDSMITH_DRIVE_PATH", _DEFAULT_DRIVE))


def register_chief_examiner(app: FastMCP):
    @app.tool(
        title="Run Chief Examiner",
        description=(
            "Run the Chief Examiner diagnostic for one iteration. Reads the feedback "
            "parquet and evaluation metrics from disk, loads iteration history from "
            "prior proposals and metrics files, and produces a DiagnosticReport + "
            "HumanReviewProposal saved to disk. Returns the proposal for human review. "
            "Call approve_proposal or reject_proposal after the human decides."
        ),
    )
    async def run_chief_examiner(
        session_id: str,
        iteration: int,
        critique: str | None = None,
    ) -> dict:
        drive_path = _drive_path()
        state = load_state(drive_path, session_id)

        feedback_path = drive_path / "sessions" / session_id / f"feedback_iter{iteration}.parquet"
        if not feedback_path.exists():
            return {
                "error": f"Feedback parquet not found at {feedback_path}. "
                         "Run run_examiner_pass first."
            }

        feedback_df = pd.read_parquet(feedback_path)

        try:
            metrics = load_metrics(drive_path, session_id, iteration)
            val_metrics = metrics.val
            test_metrics_summary = metrics.test
        except FileNotFoundError:
            return {
                "error": f"Metrics file not found for iteration {iteration}. "
                         "Run evaluate_scorer first."
            }

        provider = OpenRouterProvider()

        _, proposal = await run_diagnostic(
            session_id=session_id,
            iteration=iteration,
            feedback_df=feedback_df,
            val_metrics=val_metrics,
            test_metrics_summary=test_metrics_summary,
            policies=state.policies,
            strategy=state.strategy_guidance,
            provider=provider,
            model_config=state.models,
            drive_path=drive_path,
            critique=critique,
        )

        save_proposal(proposal, drive_path)

        return {
            "session_id": session_id,
            "iteration": iteration,
            "status": proposal.status,
            "diagnostic_summary": proposal.diagnostic_report.summary,
            "per_component_issues": proposal.diagnostic_report.per_component_issues,
            "proposed_strategy": proposal.proposed_strategy.model_dump(),
            "proposed_policies": {
                k: v.model_dump() for k, v in proposal.proposed_policies.items()
            },
            "proposal_path": str(
                drive_path / "sessions" / session_id / "proposals" / f"iter{iteration}.json"
            ),
        }

    return run_chief_examiner


def register_approve_proposal(app: FastMCP):
    @app.tool(
        title="Approve Proposal",
        description=(
            "Apply the Chief Examiner's proposed policies and strategy to SessionState "
            "and increment the iteration counter. Marks the proposal as approved on disk. "
            "Call after the human has reviewed and accepted the DiagnosticReport."
        ),
    )
    def approve_proposal(session_id: str, iteration: int) -> dict:
        drive_path = _drive_path()
        proposal = load_proposal(drive_path, session_id, iteration)
        state = load_state(drive_path, session_id)

        state.policies = proposal.proposed_policies
        state.strategy_guidance = proposal.proposed_strategy
        state.iteration = iteration + 1

        proposal.status = "approved"

        save_state(state, drive_path)
        save_proposal(proposal, drive_path)

        return {
            "session_id": session_id,
            "approved_iteration": iteration,
            "next_iteration": state.iteration,
            "updated_policies": {k: v.model_dump() for k, v in state.policies.items()},
            "updated_strategy": state.strategy_guidance.model_dump(),
        }

    return approve_proposal


def register_reject_proposal(app: FastMCP):
    @app.tool(
        title="Reject Proposal",
        description=(
            "Mark the Chief Examiner's proposal as rejected and store the human's critique. "
            "SessionState is not modified — policies and strategy remain unchanged. "
            "Re-run run_chief_examiner with the stored critique to get a revised proposal."
        ),
    )
    def reject_proposal(session_id: str, iteration: int, critique: str) -> dict:
        drive_path = _drive_path()
        proposal = load_proposal(drive_path, session_id, iteration)

        proposal.status = "rejected"
        proposal.critique = critique

        save_proposal(proposal, drive_path)

        return {
            "session_id": session_id,
            "rejected_iteration": iteration,
            "critique_stored": critique,
            "next_step": (
                "Call run_chief_examiner(session_id, iteration, critique=...) "
                "to get a revised proposal incorporating this critique."
            ),
        }

    return reject_proposal
