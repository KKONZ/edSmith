from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
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
from edsmith.providers.base import LLMProvider, Message
from edsmith.session.state import load_metrics, load_proposal

_AUDIT_SAMPLE_SIZE = 10


# ---------------------------------------------------------------------------
# Iteration history
# ---------------------------------------------------------------------------

def _load_iteration_history(
    drive_path: Path,
    session_id: str,
    current_iteration: int,
) -> str:
    """Build a narrative of prior iterations from proposals and metrics files."""
    if current_iteration == 0:
        return "No prior iterations — this is the first diagnostic."

    lines: list[str] = []

    for i in range(current_iteration):
        parts: list[str] = [f"**Iteration {i}**"]

        try:
            m = load_metrics(drive_path, session_id, i)
            parts.append(
                f"val accuracy={m.val.get('accuracy', '?'):.3f} "
                f"qwk={m.val.get('qwk', '?'):.3f} "
                f"smd={m.val.get('smd', '?'):+.3f} "
                f"adj_acc={m.val.get('adjacent_accuracy', '?'):.3f}"
            )
        except FileNotFoundError:
            parts.append("metrics not available")

        try:
            proposal = load_proposal(drive_path, session_id, i)
            report = proposal.diagnostic_report

            if report.summary:
                parts.append(f"diagnosis: {report.summary[:300]}")

            changes: list[str] = []
            sg = proposal.proposed_strategy
            for flag in ("use_aoa", "use_grammar", "use_complexity", "use_discourse",
                         "contrastive_anchoring"):
                if getattr(sg, flag):
                    changes.append(f"{flag}=True")
            for comp, focus in sg.per_component_focus.items():
                changes.append(f"{comp}_focus='{focus[:60]}'")
            for comp, pol in proposal.proposed_policies.items():
                changes.append(f"{comp}.specificity={pol.specificity}")
                if pol.additional_instructions:
                    changes.append(f"{comp}.instructions='{pol.additional_instructions[:60]}'")
            if changes:
                parts.append(f"proposed: {', '.join(changes)}")

            if proposal.status == "approved":
                parts.append("human: approved")
            elif proposal.status == "rejected":
                critique = proposal.critique or "(no critique)"
                parts.append(f"human: rejected — {critique[:200]}")
            else:
                parts.append("human: pending")

        except FileNotFoundError:
            pass

        lines.append(" | ".join(parts))

    return "## Iteration History\n\n" + "\n\n".join(lines)


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
            aoa_vals = [r["stats"]["mean_aoa_of_errors"] for r in results
                        if "mean_aoa_of_errors" in r["stats"]]
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

def _feedback_sample(feedback_df: pd.DataFrame, n: int = 12) -> str:
    """Sample individual feedback records sorted by largest score-band divergence.

    Each record shows predicted score, reference band, delta, and confidence so
    the Chief Examiner can identify specific mismatches between what the Examiner
    decided and what the reference label says.
    """
    per_component = max(2, n // len(COMPONENT_HEADINGS))
    lines: list[str] = []

    for component in COMPONENT_HEADINGS:
        subset = feedback_df[feedback_df["component"] == component].copy()
        if subset.empty:
            continue

        try:
            scores = pd.to_numeric(subset["score"], errors="coerce")
            bands = pd.to_numeric(subset["band"], errors="coerce")
            subset = subset.assign(_delta=(scores - bands).abs())
            subset = subset.sort_values("_delta", ascending=False)
        except Exception:
            pass

        for _, row in subset.head(per_component).iterrows():
            predicted = row.get("score", "?")
            reference = row.get("band", "?")
            tag = row.get("tag", "")
            text = str(row.get("feedback_text", ""))[:400]

            delta_str = ""
            try:
                delta = float(predicted) - float(reference)
                delta_str = f" Δ={delta:+.1f}"
            except (TypeError, ValueError):
                pass

            tag_str = f" confidence={tag}" if tag else ""
            lines.append(
                f"[{component} | predicted={predicted} reference={reference}{delta_str}{tag_str}]\n{text}"
            )

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
    iteration_history: str,
    critique: str | None,
    trained_components: list[str] | None = None,
) -> list[Message]:
    system = (
        "You are the Chief Examiner for an IELTS automated scoring system. "
        "Your role is to diagnose feedback quality issues in the training data and "
        "propose targeted changes to PromptPolicy and StrategyGuidance for the next iteration.\n\n"
        "Use the iteration history to avoid repeating changes that did not help and to "
        "build on changes that did. If a human has rejected a prior proposal with a critique, "
        "that critique must directly shape your new proposal.\n\n"
        "Test set purity rule: you may inspect individual validation records but NEVER "
        "individual test records. Only aggregated test statistics are provided.\n\n"
        "Respond with a single JSON object inside <diagnostic>...</diagnostic> tags.\n"
        "Schema:\n"
        "{\n"
        '  "summary": "concise diagnosis of the main issue this iteration",\n'
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
        "Only include components in proposed_policies where you recommend changes. "
        "Focus on the highest-leverage fix — do not change more than 2–3 things at once."
    )

    parts = [
        f"## Session {session_id} — Iteration {iteration}",
    ]
    if trained_components:
        parts.append(
            f"\n### Trained Components This Iteration\n"
            f"Only these components had a Scorer trained and evaluated: {', '.join(trained_components)}. "
            f"Metrics and per_component_issues should focus on these components only. "
            f"Do not diagnose or propose policy changes for untrained components."
        )
    parts += [
        f"\n{iteration_history}",
        "\n### Current Validation Metrics",
        json.dumps(val_metrics, indent=2),
        "\n### Test Metrics (aggregated summary only — no individual records)",
        json.dumps(test_metrics_summary, indent=2),
        "\n### Current Policies",
        json.dumps({k: v.model_dump() for k, v in policies.items()}, indent=2),
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

    strategy_data = strategy.model_dump()
    strategy_data.update(raw.get("proposed_strategy", {}))
    try:
        proposed_strategy = StrategyGuidance.model_validate(strategy_data)
    except Exception:
        proposed_strategy = strategy

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
    drive_path: Path,
    critique: str | None = None,
    trained_components: list[str] | None = None,
) -> tuple[DiagnosticReport, HumanReviewProposal]:
    """Diagnose feedback quality and produce a HumanReviewProposal.

    Loads iteration history from prior proposals and metrics files so the
    Chief Examiner can reason about what has been tried and what the human
    has approved or rejected.

    Individual test records are never passed here — only aggregated
    test_metrics_summary, consistent with ADR 0006.
    """
    iteration_history = _load_iteration_history(drive_path, session_id, iteration)

    sample_essays = (
        feedback_df["essay"].drop_duplicates().head(_AUDIT_SAMPLE_SIZE).tolist()
    )
    linguistic_findings = await asyncio.to_thread(
        _linguistic_audit_sync, sample_essays, strategy
    )

    fb_sample = _feedback_sample(feedback_df)
    _skip = {"confusion_matrix"}
    metric_summary = {
        **{k: v for k, v in val_metrics.items() if k not in _skip},
        **{f"test_{k}": v for k, v in test_metrics_summary.items() if k not in _skip},
    }

    messages = _build_messages(
        session_id=session_id,
        iteration=iteration,
        val_metrics=val_metrics,
        test_metrics_summary=test_metrics_summary,
        policies=policies,
        strategy=strategy,
        linguistic_findings=linguistic_findings,
        feedback_sample=fb_sample,
        iteration_history=iteration_history,
        critique=critique,
        trained_components=trained_components,
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
