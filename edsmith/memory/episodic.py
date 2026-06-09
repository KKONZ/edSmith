from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# Sub-models
# ------------------------------------------------------------------

class IterationMetrics(BaseModel):
    iteration: int
    accuracy: float
    adjacent_accuracy: float
    qwk: float
    smd: float


class PolicySnapshot(BaseModel):
    specificity: int
    evidence_required: bool
    feedback_granularity: str
    additional_instructions: str


# ------------------------------------------------------------------
# Core record
# ------------------------------------------------------------------

class EpisodicRecord(BaseModel):
    # Identity
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_session_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Tree navigation
    tree_depth: int = 0
    action_taken: str = ""

    # MCTS / UCB1 fields — updated in-place as the tree is searched
    visit_count: int = 0
    value_estimate: float = 0.0      # tracks best accuracy seen at this node

    # Architecture
    architecture: Literal["react", "council", "hybrid"] = "react"
    council_enabled: bool = False
    council_chair_memory_injection: bool = False

    # Config snapshot (prompt policies at session start)
    prompt_policies: dict[str, PolicySnapshot] = Field(default_factory=dict)

    # Per-iteration results (appended as iterations complete)
    iterations: list[IterationMetrics] = Field(default_factory=list)

    # Natural language reflection body
    reflection_notes: str = ""

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def best_accuracy(self) -> float | None:
        if not self.iterations:
            return None
        return max(m.accuracy for m in self.iterations)

    @property
    def final_metrics(self) -> IterationMetrics | None:
        return self.iterations[-1] if self.iterations else None


# ------------------------------------------------------------------
# Storage (one Markdown file per session)
# ------------------------------------------------------------------

_FRONTMATTER_SEP = "---"


def _serialize(record: EpisodicRecord) -> str:
    data = record.model_dump(mode="json")
    body = data.pop("reflection_notes", "")
    frontmatter = yaml.dump(data, default_flow_style=False, allow_unicode=True)
    return f"{_FRONTMATTER_SEP}\n{frontmatter}{_FRONTMATTER_SEP}\n\n{body}"


def _deserialize(text: str) -> EpisodicRecord:
    parts = text.split(_FRONTMATTER_SEP, 2)
    if len(parts) < 3:
        raise ValueError("Invalid episodic record: missing frontmatter delimiters")
    data = yaml.safe_load(parts[1])
    data["reflection_notes"] = parts[2].strip()
    return EpisodicRecord.model_validate(data)


# ------------------------------------------------------------------
# EpisodicMemory — read/write facade
# ------------------------------------------------------------------

class EpisodicMemory:
    """File-backed episodic memory tree.

    Each session is stored as a Markdown file with YAML frontmatter under
    `{drive_path}/episodic/`.  Tree structure is encoded via parent_session_id
    and tree_depth; no separate index is maintained.
    """

    def __init__(self, drive_path: str | Path) -> None:
        self._dir = Path(drive_path) / "episodic"
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self._dir / f"{session_id}.md"

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, record: EpisodicRecord) -> None:
        self._path(record.session_id).write_text(_serialize(record), encoding="utf-8")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load(self, session_id: str) -> EpisodicRecord:
        path = self._path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"No episodic record for session {session_id!r}")
        return _deserialize(path.read_text(encoding="utf-8"))

    def load_all(self) -> list[EpisodicRecord]:
        records = []
        for p in sorted(self._dir.glob("*.md")):
            try:
                records.append(_deserialize(p.read_text(encoding="utf-8")))
            except Exception:
                continue
        return records

    # ------------------------------------------------------------------
    # Tree queries
    # ------------------------------------------------------------------

    def get_children(self, session_id: str) -> list[EpisodicRecord]:
        return [r for r in self.load_all() if r.parent_session_id == session_id]

    def get_siblings(self, session_id: str) -> list[EpisodicRecord]:
        """Records that share the same parent (excludes the node itself)."""
        record = self.load(session_id)
        if record.parent_session_id is None:
            return []
        return [
            r for r in self.load_all()
            if r.parent_session_id == record.parent_session_id
            and r.session_id != session_id
        ]

    def get_nodes_at_depth(self, depth: int) -> list[EpisodicRecord]:
        return [r for r in self.load_all() if r.tree_depth == depth]

    def beam_search_eligible(self, session_id: str) -> bool:
        """True when ≥2 sibling nodes exist (ADR-0003 threshold)."""
        return len(self.get_siblings(session_id)) >= 2

    def mcts_eligible(self, min_nodes: int = 5) -> bool:
        """True when the tree has enough nodes for meaningful UCB1 selection."""
        return len(list(self._dir.glob("*.md"))) >= min_nodes
