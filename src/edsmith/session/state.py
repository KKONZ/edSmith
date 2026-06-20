from __future__ import annotations

import json
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from edsmith.config.session import HumanReviewProposal, PromptPolicy, StrategyGuidance


class SessionState(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    iteration: int = 0
    policies: dict[str, PromptPolicy] = Field(default_factory=dict)
    strategy_guidance: StrategyGuidance = Field(default_factory=StrategyGuidance)
    model_path: str | None = None
    parent_session_id: str | None = None


class SessionMetrics(BaseModel):
    """Evaluation metrics for one iteration, persisted after evaluate_scorer returns.

    test metrics are aggregated summaries only — no individual test records
    are stored here (ADR 0006).
    """
    session_id: str
    iteration: int
    val: dict[str, float]
    test: dict[str, float]


def _state_path(drive_path: Path, session_id: str) -> Path:
    return drive_path / "sessions" / session_id / "state.json"


def load_state(drive_path: Path, session_id: str) -> SessionState:
    path = _state_path(drive_path, session_id)
    return SessionState.model_validate_json(path.read_text())


def save_state(state: SessionState, drive_path: Path) -> Path:
    path = _state_path(drive_path, state.session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.model_dump_json(indent=2))
    return path


def _metrics_path(drive_path: Path, session_id: str, iteration: int) -> Path:
    return drive_path / "sessions" / session_id / f"metrics_iter{iteration}.json"


def save_metrics(metrics: SessionMetrics, drive_path: Path) -> Path:
    path = _metrics_path(drive_path, metrics.session_id, metrics.iteration)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(metrics.model_dump_json(indent=2))
    return path


def load_metrics(drive_path: Path, session_id: str, iteration: int) -> SessionMetrics:
    path = _metrics_path(drive_path, session_id, iteration)
    return SessionMetrics.model_validate_json(path.read_text())


def _proposal_path(drive_path: Path, session_id: str, iteration: int) -> Path:
    return drive_path / "sessions" / session_id / "proposals" / f"iter{iteration}.json"


def load_proposal(drive_path: Path, session_id: str, iteration: int) -> HumanReviewProposal:
    path = _proposal_path(drive_path, session_id, iteration)
    return HumanReviewProposal.model_validate_json(path.read_text())


def save_proposal(proposal: HumanReviewProposal, drive_path: Path) -> Path:
    path = _proposal_path(drive_path, proposal.session_id, proposal.iteration)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(proposal.model_dump_json(indent=2))
    return path
