from __future__ import annotations

import asyncio
import json
import math
import re
from typing import Any

import pandas as pd

from edsmith.config.session import (
    DiagnosticReport,
    HumanReviewProposal,
    ModelConfig,
    PromptPolicy,
    StrategyGuidance,
)
from edsmith.data.parser import COMPONENT_HEADINGS
from edsmith.memory.episodic import EpisodicMemory, EpisodicRecord
from edsmith.providers.base import LLMProvider, Message

_UCB1_C = math.sqrt(2)
_AUDIT_SAMPLE_SIZE = 10


# ---------------------------------------------------------------------------
# UCB1
# ---------------------------------------------------------------------------

def _ucb1(record: EpisodicRecord, total_visits: int) -> float:
    if record.visit_count == 0:
        return float("inf")
    return record.value_estimate + _UCB1_C * math.sqrt(
        math.log(total_visits) / record.visit_count
    )


# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------

def _select_mode(session_id: str, memory: EpisodicMemory) -> tuple[str, str]:
    """Return (mode, context_text) for the diagnostic prompt."""
    all_records = memory.load_all()
    total_nodes = len(all_records)
    siblings = memory.get_siblings(session_id)

    if total_nodes >= 5:
        mode = "mcts"
    elif len(siblings) >= 2:
        mode = "beam"
    else:
        mode = "simple"

    lines = [f"Reflection mode: {mode} ({total_nodes} sessions in tree, {len(siblings)} siblings)."]

    if mode in ("beam", "mcts") and siblings:
        lines.append("Sibling session performance:")
        for sib in siblings[:3]:
            m = sib.final_metrics
            if m:
                lines.append(
                    f"  {sib.session_id}: accuracy={m.accuracy:.3f}, "
                    f"qwk={m.qwk:.3f}, smd={m.smd:+.3f}"
                )

    if mode == "mcts":
        total_visits = sum(r.visit_count for r in all_records)
        if total_visits > 0:
            candidates = [r for r in all_records if r.session_id != session_id]
            if candidates:
                best = max(candidates, key=lambda r: _ucb1(r, total_visits))
                lines.append(
                    f"MCTS: highest UCB1 node is {best.session_id} "
                    f"(score={_ucb1(best, total_visits):.3f}, "
                    f"value={best.value_estimate:.3f}, visits={best.visit_count})"
                )

    return mode, "\n".join(lines)


# ---------------------------------------------------------------------------
# Linguistic audit (synchronous — called via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _linguistic_audit_sync(
    essays: list[str],
    strategy: StrategyGuidance,
) -> dict[str, Any]:
    findings: dict[str, Any] = {}

    if strategy.use_grammar:
        try:
            from edsmith.tools.grammar import grammar_check
            results = [grammar_check(e) for e in essays]
            aoa_vals = [r["stats"].get("mean_aoa_of_errors") for r in results
                        if r["stats"].get("mean_aoa_of_errors") is not None]
            findings["grammar"] = {
                "mean_errors_per_essay": sum(r["count"] for r in results) / len(results),
                "mean_aoa_of_errors": sum(aoa_vals) / len(aoa_vals) if aoa_vals else None,
                "pct_errors_basic": sum(r["stats"].get("pct_errors_basic", 0) for r in results) / len(results),
            }
        except ImportError:
            pass

    if strategy.use_aoa:
        try:
            from edsmith.tools.aoa import compute_aoa_stats
            results = [compute_aoa_stats(e) for e in essays]
            findings["aoa"] = {
                "mean_aoa": sum(r["stats"].get("aoa_mean", 0) for r in results) / len(results),
                "mean_pct_late": sum(r["stats"].get("pct_late", 0) for r in results) / len(results),
                "mean_aoa_std": sum(r["stats"].get("aoa_std", 0) for r in results) / len(results),
            }
        except ImportError:
            pass

    if strategy.use_complexity:
        try:
            from edsmith.tools.complexity import complexity_stats
            results = [complexity_stats(e) for e in essays]
            findings["complexity"] = {
                "mean_passive_ratio": sum(r["stats"].get("passive_ratio", 0) for r in results) / len(results),
                "mean_subordinate_ratio": sum(r["stats"].get("subordinate_ratio", 0) for r in results) / len(results),
                "mean_nominalization_ratio": sum(r["stats"].get("nominalization_ratio", 0) for r in results) / len(results),
                "mean_dep_depth": sum(r["stats"].get("dep_depth_mean", 0) for r in results) / len(results),
            }
        except ImportError:
            pass

    if strategy.use_discourse:
        try:
            from edsmith.tools.discourse import discourse_analysis
            results = [discourse_analysis(e) for e in essays]
            findings["discourse"] = {
                "mean_transitions": sum(r["stats"].get("total_transitions_wordlist", 0) for r in results) / len(results),
                "mean_pronoun_ratio": sum(r["stats"].get("pronoun_ratio", 0) for r in results) / len(results),
                "mean_repetition_rate": sum(r["stats"].get("lexical_repetition_rate", 0) for r in results) / len(results),
                "pct_with_intro_marker": sum(r["stats"].get("has_introduction_marker", 0) for r in results) / len(results),
                "pct_with_conclusion_marker": sum(r["stats"].get("has_conclusion_marker", 0) for r in results) / len(results),
            }
        except ImportError:
            pass

    return findings


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _feedback_sample(feedback_df: pd.DataFrame, n: int = 6) -> str:
    """Return a readable sample of feedback text for the diagnostic prompt."""
    lines = []
    for component in COMPONENT_HEADINGS:
        subset = feedback_df[feedback_df["component"] == component].head(n // 4 or 1)
        for _, row in subset.iterrows():
            score = row.get("score", "?")
            text = str(row.get("feedback_text", ""))[:300]
            lines.append(f"[{component} | score={score}] {text}")
    return "\n\n".join(lines)


def _build_messages(
    session_id: str,
    iteration: int,
    val_metrics: dict[str, float],
    test_metrics_summary: dict[str, float],
    policies: dict[str, PromptPolicy],
    strategy: StrategyGuidance,
    linguistic_findings: dict[str, Any],
    feedback_sample: str,
    mode_context: str,
    critique: str | None,
) -> list[Message]:
    system = (
        "You are the Chief Examiner for an IELTS automated scoring system. "
        "Your role is to diagnose feedback quality issues in the training data and "
        "propose targeted changes to PromptPolicy and StrategyGuidance for the next iteration.\n\n"
        "Test set purity rule: you may inspect individual validation records but NEVER "
        "individual test records. Only aggregated test statistics are provided.\n\n"
        "Respond with a single JSON object inside <diagnostic>...</diagnostic> tags.\n"
        "Schema:\n"
        "{\n"
        '  "summary": "...",\n'
        '  "per_component_issues": {"task_response": "...", "coherence": "...", "lexical": "...", "grammar": "..."},\n'
        '  "proposed_strategy": {"use_aoa": bool, "use_grammar": bool, "use_complexity": bool,\n'
        '                        "use_discourse": bool, "contrastive_anchoring": bool,\n'
        '                        "per_component_focus": {"component": "focus instruction"}},\n'
        '  "proposed_policies": {\n'
        '    "component": {"specificity": 1-5, "evidence_required": bool,\n'
        '                  "feedback_granularity": "component|overall|both",\n'
        '                  "additional_instructions": "..."}\n'
        '  }\n'
        "}\n"
        "Only include components in proposed_policies where you recommend changes."
    )

    parts = [
        f"## Session {session_id} — Iteration {iteration}",
        f"\n### {mode_context}",
        "\n### Validation Metrics",
        json.dumps(val_metrics, indent=2),
        "\n### Test Metrics (summary only — no individual records)",
        json.dumps(test_metrics_summary, indent=2),
        "\n### Current Policies",
        json.dumps(
            {k: v.model_dump() for k, v in policies.items()},
            indent=2,
        ),
        "\n### Current Strategy",
        strategy.model_dump_json(indent=2),
    ]

    if linguistic_findings:
        parts.append("\n### Linguistic Audit Findings (sampled essays)")
        parts.append(json.dumps(linguistic_findings, indent=2))

    if feedback_sample:
        parts.append("\n### Feedback Sample (validation set)")
        parts.append(feedback_sample)

    if critique:
        parts.append(f"\n### Human Critique of Previous Proposal\n{critique}")

    parts.append(
        "\nDiagnose the feedback quality and propose changes. "
        "Focus on the highest-leverage fix — do not change more than 2–3 things at once."
    )

    return [
        Message(role="system", content=system),
        Message(role="user", content="\n".join(parts)),
    ]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_response(
    text: str,
    session_id: str,
    iteration: int,
    policies: dict[str, PromptPolicy],
    strategy: StrategyGuidance,
    linguistic_findings: dict[str, Any],
    metric_summary: dict[str, float],
) -> tuple[DiagnosticReport, HumanReviewProposal]:
    raw: dict[str, Any] = {}

    m = re.search(r"<diagnostic>(.*?)</diagnostic>", text, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            raw = json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    report = DiagnosticReport(
        session_id=session_id,
        iteration=iteration,
        summary=raw.get("summary", ""),
        per_component_issues=raw.get("per_component_issues", {}),
        metric_summary=metric_summary,
        linguistic_findings=linguistic_findings,
    )

    # Merge proposed strategy — start from current and overlay changes
    strategy_data = strategy.model_dump()
    strategy_data.update(raw.get("proposed_strategy", {}))
    try:
        proposed_strategy = StrategyGuidance.model_validate(strategy_data)
    except Exception:
        proposed_strategy = strategy

    # Merge proposed policies — start from current and overlay per-component changes
    proposed_policies: dict[str, PromptPolicy] = dict(policies)
    for component, patch in raw.get("proposed_policies", {}).items():
        if component in COMPONENT_HEADINGS:
            base = policies.get(component, PromptPolicy()).model_dump()
            base.update(patch)
            try:
                proposed_policies[component] = PromptPolicy.model_validate(base)
            except Exception:
                pass

    proposal = HumanReviewProposal(
        session_id=session_id,
        iteration=iteration,
        diagnostic_report=report,
        proposed_strategy=proposed_strategy,
        proposed_policies=proposed_policies,
    )

    return report, proposal


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_diagnostic(
    session_id: str,
    iteration: int,
    feedback_df: pd.DataFrame,
    val_metrics: dict[str, float],
    test_metrics_summary: dict[str, float],
    policies: dict[str, PromptPolicy],
    strategy: StrategyGuidance,
    provider: LLMProvider,
    model_config: ModelConfig,
    episodic_memory: EpisodicMemory,
    critique: str | None = None,
) -> tuple[DiagnosticReport, HumanReviewProposal]:
    """Diagnose feedback quality and produce a HumanReviewProposal.

    Individual test records are never passed here — only aggregated
    test_metrics_summary, consistent with ADR 0006.
    """
    mode, mode_context = _select_mode(session_id, episodic_memory)

    # Linguistic audit on a sample of essays (synchronous tools → thread)
    sample_essays = (
        feedback_df["essay"]
        .drop_duplicates()
        .head(_AUDIT_SAMPLE_SIZE)
        .tolist()
    )
    linguistic_findings = await asyncio.to_thread(
        _linguistic_audit_sync, sample_essays, strategy
    )

    fb_sample = _feedback_sample(feedback_df)
    metric_summary = {**val_metrics, **{f"test_{k}": v for k, v in test_metrics_summary.items()}}

    messages = _build_messages(
        session_id=session_id,
        iteration=iteration,
        val_metrics=val_metrics,
        test_metrics_summary=test_metrics_summary,
        policies=policies,
        strategy=strategy,
        linguistic_findings=linguistic_findings,
        feedback_sample=fb_sample,
        mode_context=mode_context,
        critique=critique,
    )

    response = await provider.acomplete(messages, model=model_config.chair)
    return _parse_response(
        text=response.content,
        session_id=session_id,
        iteration=iteration,
        policies=policies,
        strategy=strategy,
        linguistic_findings=linguistic_findings,
        metric_summary=metric_summary,
    )
