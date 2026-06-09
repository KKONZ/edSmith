from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Literal

from edsmith.config.session import PromptPolicy
from edsmith.data.parser import COMPONENT_HEADINGS
from edsmith.memory.episodic import EpisodicMemory, EpisodicRecord
from edsmith.providers.base import LLMProvider, Message


_UCB1_C = math.sqrt(2)  # standard exploration constant

ReflectionMode = Literal["simple", "beam", "mcts"]


# ------------------------------------------------------------------
# Output
# ------------------------------------------------------------------

@dataclass
class ReflectionOutput:
    mode: ReflectionMode
    notes: str
    action_taken: str
    suggested_policies: dict[str, PromptPolicy]
    input_tokens: int
    output_tokens: int


# ------------------------------------------------------------------
# System prompt
# ------------------------------------------------------------------

_SYSTEM = """\
You are an AI research assistant guiding an iterative IELTS automated-scoring experiment.

Each session runs N fine-tuning iterations. In each iteration:
  Phase 1 — a feedback agent generates per-component feedback for a batch of essays.
  Phase 2 — a Scorer model is fine-tuned on that feedback, then evaluated.

Metrics tracked per iteration (validation set):
  accuracy         — exact match to human band score
  adjacent_accuracy — within one band step of human score
  qwk              — quadratic weighted kappa
  smd              — standardised mean difference (positive = over-prediction)

Primary optimisation target: ACCURACY.

The four essay components are: task_response, coherence, lexical, grammar.

Each component has a PromptPolicy that controls how feedback is generated:
  specificity (1–5)            — level of detail in the feedback
  evidence_required (bool)     — whether the agent must cite essay text
  feedback_granularity         — "component" | "overall" | "both"
  additional_instructions (str) — free-form extra guidance

Your job: analyse the session history and suggest targeted PromptPolicy changes \
that are likely to improve Scorer accuracy in the next session.

## Output Format
<notes>
Your analysis: what worked, what did not, and why.
</notes>
<action>
One sentence describing the change you are recommending.
</action>
<policies>
JSON object with partial PromptPolicy updates. Only include components and fields \
you want to change. Example:
{
  "task_response": {"specificity": 4},
  "coherence": {"evidence_required": false}
}
</policies>"""


# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------

class ReflectionAgent:
    """Phase 2 reflection agent.

    Selects a reflection mode based on the episodic tree, assembles the
    appropriate context, and asks the LLM to recommend PromptPolicy changes
    for the next session.

    Mode escalation:
      simple — fewer than 2 siblings and fewer than 5 total tree nodes
      beam   — ≥2 sibling sessions exist (compare branches, pick best direction)
      mcts   — ≥5 total tree nodes (UCB1 node selection guides exploration)

    Test-set purity: only aggregated summary statistics are ever passed in;
    individual test records are never exposed here.
    """

    def __init__(
        self,
        provider: LLMProvider,
        episodic_memory: EpisodicMemory,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> None:
        self._provider = provider
        self._episodic = episodic_memory
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reflect(
        self,
        record: EpisodicRecord,
        current_policies: dict[str, PromptPolicy],
        val_summary: dict[str, float],
        test_summary: dict[str, float] | None = None,
    ) -> ReflectionOutput:
        mode = self._select_mode(record)
        messages = self._build_messages(mode, record, current_policies, val_summary, test_summary)
        resp = self._provider.complete(messages, self._model, self._temperature, self._max_tokens)
        return _parse_response(mode, resp.content, current_policies, resp.input_tokens, resp.output_tokens)

    async def areflect(
        self,
        record: EpisodicRecord,
        current_policies: dict[str, PromptPolicy],
        val_summary: dict[str, float],
        test_summary: dict[str, float] | None = None,
    ) -> ReflectionOutput:
        mode = self._select_mode(record)
        messages = self._build_messages(mode, record, current_policies, val_summary, test_summary)
        resp = await self._provider.acomplete(messages, self._model, self._temperature, self._max_tokens)
        return _parse_response(mode, resp.content, current_policies, resp.input_tokens, resp.output_tokens)

    # ------------------------------------------------------------------
    # Mode selection
    # ------------------------------------------------------------------

    def _select_mode(self, record: EpisodicRecord) -> ReflectionMode:
        if self._episodic.mcts_eligible():
            return "mcts"
        if self._episodic.beam_search_eligible(record.session_id):
            return "beam"
        return "simple"

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        mode: ReflectionMode,
        record: EpisodicRecord,
        current_policies: dict[str, PromptPolicy],
        val_summary: dict[str, float],
        test_summary: dict[str, float] | None,
    ) -> list[Message]:
        if mode == "simple":
            user = _simple_user(record, current_policies, val_summary, test_summary)
        elif mode == "beam":
            siblings = self._episodic.get_siblings(record.session_id)
            user = _beam_user(record, current_policies, val_summary, test_summary, siblings)
        else:  # mcts
            all_records = self._episodic.load_all()
            user = _mcts_user(record, current_policies, val_summary, test_summary, all_records)
        return [Message(role="system", content=_SYSTEM), Message(role="user", content=user)]


# ------------------------------------------------------------------
# User message builders
# ------------------------------------------------------------------

def _simple_user(
    record: EpisodicRecord,
    policies: dict[str, PromptPolicy],
    val_summary: dict[str, float],
    test_summary: dict[str, float] | None,
) -> str:
    return "\n\n".join([
        f"## Session {record.session_id}",
        _session_header(record),
        _policies_block(policies),
        _iteration_table(record),
        _summary_block("Validation Summary", val_summary),
        _summary_block("Test Summary (aggregated)", test_summary) if test_summary else "",
        "Reflect on these results and suggest PromptPolicy changes for the next session.",
    ]).strip()


def _beam_user(
    record: EpisodicRecord,
    policies: dict[str, PromptPolicy],
    val_summary: dict[str, float],
    test_summary: dict[str, float] | None,
    siblings: list[EpisodicRecord],
) -> str:
    sibling_blocks = []
    for sib in sorted(siblings, key=lambda r: r.best_accuracy or 0.0, reverse=True):
        sibling_blocks.append(
            f"### Sibling {sib.session_id}  "
            f"(best accuracy: {sib.best_accuracy:.4f})\n"
            + _iteration_table(sib)
        )

    return "\n\n".join([
        f"## Current Session {record.session_id}",
        _session_header(record),
        _policies_block(policies),
        _iteration_table(record),
        _summary_block("Validation Summary", val_summary),
        _summary_block("Test Summary (aggregated)", test_summary) if test_summary else "",
        "## Sibling Sessions (same parent)",
        "\n\n".join(sibling_blocks) if sibling_blocks else "None.",
        (
            "Compare the current session against its siblings. "
            "Identify the most promising direction and recommend PromptPolicy changes."
        ),
    ]).strip()


def _mcts_user(
    record: EpisodicRecord,
    policies: dict[str, PromptPolicy],
    val_summary: dict[str, float],
    test_summary: dict[str, float] | None,
    all_records: list[EpisodicRecord],
) -> str:
    total_visits = sum(r.visit_count for r in all_records) or 1
    scored = sorted(
        [(r, _ucb1(r, total_visits)) for r in all_records],
        key=lambda x: x[1],
        reverse=True,
    )
    tree_lines = ["| session_id | depth | visits | best_acc | UCB1 |", "|---|---|---|---|---|"]
    for r, score in scored:
        acc = f"{r.best_accuracy:.4f}" if r.best_accuracy is not None else "—"
        ucb = "∞" if score == float("inf") else f"{score:.4f}"
        marker = " ◀ current" if r.session_id == record.session_id else ""
        tree_lines.append(f"| {r.session_id}{marker} | {r.tree_depth} | {r.visit_count} | {acc} | {ucb} |")

    selected = scored[0][0]
    selected_note = (
        f"Node **{selected.session_id}** has the highest UCB1 score "
        f"({'current session' if selected.session_id == record.session_id else 'a different branch'})."
    )

    return "\n\n".join([
        f"## Current Session {record.session_id}",
        _session_header(record),
        _policies_block(policies),
        _iteration_table(record),
        _summary_block("Validation Summary", val_summary),
        _summary_block("Test Summary (aggregated)", test_summary) if test_summary else "",
        "## Full Session Tree (UCB1)",
        "\n".join(tree_lines),
        selected_note,
        (
            "Using the UCB1 analysis to balance exploration and exploitation, "
            "recommend PromptPolicy changes that extend the most promising branch."
        ),
    ]).strip()


# ------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------

def _session_header(record: EpisodicRecord) -> str:
    return (
        f"Architecture: {record.architecture}  |  "
        f"Depth: {record.tree_depth}  |  "
        f"Parent: {record.parent_session_id or 'root'}  |  "
        f"Best accuracy: {record.best_accuracy:.4f}"
        if record.best_accuracy is not None
        else f"Architecture: {record.architecture}  |  Depth: {record.tree_depth}  |  No iterations yet"
    )


def _policies_block(policies: dict[str, PromptPolicy]) -> str:
    lines = ["## Current Prompt Policies", "| component | specificity | evidence_required | granularity | extra |", "|---|---|---|---|---|"]
    for comp, p in policies.items():
        extra = p.additional_instructions[:40] + "…" if len(p.additional_instructions) > 40 else p.additional_instructions
        lines.append(f"| {comp} | {p.specificity} | {p.evidence_required} | {p.feedback_granularity} | {extra} |")
    return "\n".join(lines)


def _iteration_table(record: EpisodicRecord) -> str:
    if not record.iterations:
        return "No iteration data."
    lines = ["| iter | accuracy | adj_accuracy | qwk | smd |", "|---|---|---|---|---|"]
    for m in record.iterations:
        lines.append(f"| {m.iteration} | {m.accuracy:.4f} | {m.adjacent_accuracy:.4f} | {m.qwk:.4f} | {m.smd:.4f} |")
    return "\n".join(lines)


def _summary_block(label: str, summary: dict[str, float]) -> str:
    lines = [f"## {label}"]
    for k, v in summary.items():
        lines.append(f"  {k}: {v:.4f}")
    return "\n".join(lines)


def _ucb1(record: EpisodicRecord, total_visits: int) -> float:
    if record.visit_count == 0:
        return float("inf")
    return record.value_estimate + _UCB1_C * math.sqrt(math.log(total_visits) / record.visit_count)


# ------------------------------------------------------------------
# Response parsing
# ------------------------------------------------------------------

def _parse_response(
    mode: ReflectionMode,
    content: str,
    current_policies: dict[str, PromptPolicy],
    input_tokens: int,
    output_tokens: int,
) -> ReflectionOutput:
    notes = _extract_tag(content, "notes")
    action = _extract_tag(content, "action")
    suggested = _parse_policy_updates(content, current_policies)
    return ReflectionOutput(
        mode=mode,
        notes=notes,
        action_taken=action,
        suggested_policies=suggested,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _extract_tag(content: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", content, re.DOTALL)
    return m.group(1).strip() if m else ""


def _parse_policy_updates(
    content: str,
    current: dict[str, PromptPolicy],
) -> dict[str, PromptPolicy]:
    raw = _extract_tag(content, "policies")
    if not raw:
        return dict(current)
    try:
        updates: dict = json.loads(raw)
    except json.JSONDecodeError:
        return dict(current)

    result = {}
    valid_components = set(COMPONENT_HEADINGS)
    for component, policy in current.items():
        if component in updates and component in valid_components:
            merged = policy.model_dump()
            merged.update({k: v for k, v in updates[component].items() if k in merged})
            try:
                result[component] = PromptPolicy(**merged)
            except Exception:
                result[component] = policy  # bad update — keep existing
        else:
            result[component] = policy
    return result
