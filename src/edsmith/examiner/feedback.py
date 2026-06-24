
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass

from edsmith.config.session import ModelConfig, PromptPolicy, StrategyGuidance
from edsmith.data.parser import COMPONENT_HEADINGS, _extract_score_from_body
from edsmith.examiner.rubric import get_band_descriptors
from edsmith.providers.base import LLMProvider, Message, ToolCall

logger = logging.getLogger("edsmith.examiner.feedback")

_MAX_TOOL_ROUNDS = 6


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

@dataclass
class ComponentFeedback:
    component: str
    feedback: str
    score: float | None
    tag: str | None        # LLM self-assessment, e.g. confidence level
    calibration_delta: float = 0.0  # score adjustment applied during calibration (0.0 = none)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_score(text: str) -> float | None:
    m = re.search(r"<score>([\d.]+)</score>", text, re.IGNORECASE)
    if m:
        try:
            val = float(m.group(1))
            if 0.0 <= val <= 9.0:
                return val
        except ValueError:
            pass
    return _extract_score_from_body(text)


def _extract_tag(text: str, tag: str) -> str | None:
    m = re.search(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Linguistic tool context (synchronous — run via asyncio.to_thread)
# Used only when use_tool_calling=False
# ---------------------------------------------------------------------------

def _collect_tool_context_sync(essay: str, strategy: StrategyGuidance) -> str:
    parts: list[str] = []

    if strategy.use_grammar:
        try:
            from edsmith.tools.grammar import grammar_check
            parts.append(f"Grammar analysis: {grammar_check(essay)['summary']}")
        except Exception:
            pass

    if strategy.use_aoa:
        try:
            from edsmith.tools.aoa import compute_aoa_stats
            parts.append(f"Vocabulary AoA: {compute_aoa_stats(essay)['summary']}")
        except Exception:
            pass

    if strategy.use_complexity:
        try:
            from edsmith.tools.complexity import complexity_stats
            parts.append(f"Syntactic complexity: {complexity_stats(essay)['summary']}")
        except Exception:
            pass

    if strategy.use_discourse:
        try:
            from edsmith.tools.discourse import discourse_analysis
            parts.append(f"Discourse structure: {discourse_analysis(essay)['summary']}")
        except Exception:
            pass

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_messages(
    question: str,
    essay: str,
    component: str,
    policy: PromptPolicy,
    strategy: StrategyGuidance,
    tool_context: str,
    band: float | None = None,
) -> list[Message]:
    heading = COMPONENT_HEADINGS[component]

    system_parts = [
        f"You are an expert IELTS examiner assessing the '{heading}' component.",
        f"Specificity level: {policy.specificity}/5 (1=brief overview, 5=highly detailed).",
    ]

    system_parts.append(get_band_descriptors(component))

    if policy.evidence_required:
        system_parts.append(
            "Cite specific evidence from the essay to support every point you make."
        )
    if policy.feedback_granularity in ("overall", "both"):
        system_parts.append(
            "Include both component-level observations and overall writing quality comments."
        )
    if policy.additional_instructions:
        system_parts.append(f"Additional instructions: {policy.additional_instructions}")

    focus = strategy.per_component_focus.get(component, "")
    if focus:
        system_parts.append(f"Strategic focus for this component this iteration: {focus}")
    if strategy.contrastive_anchoring:
        system_parts.append(
            "Use contrastive anchoring: explicitly compare the essay to what "
            "a higher and lower band would look like for this component."
        )

    if strategy.use_tool_calling:
        system_parts.append(
            "You have access to linguistic analysis tools. Call them on the full essay "
            "or specific excerpts to gather objective evidence before writing your assessment."
        )
        system_parts.append(
            "When using grammar and vocabulary tools together:\n"
            "- When `grammar_check` flags a word and its `aoa` field is null, the word is "
            "likely misspelled and therefore absent from the AoA vocabulary. Do NOT use the "
            "tool's replacement suggestions — they are mechanical and often miss the writer's "
            "intent. Instead, use your own language understanding to read the sentence in "
            "context and infer what the writer most likely meant to write. Then call "
            "`aoa_lookup` with that inferred word to assess its vocabulary level. Report the "
            "misspelling, your inferred intended word, and its AoA in your feedback.\n"
            "- Beyond counting surface errors, assess **comprehension impact**: does any "
            "error or pattern of errors make a sentence ambiguous or unintelligible — i.e. "
            "a reader cannot confidently reconstruct what the writer meant? Flag these "
            "explicitly. This is a more serious penalty than errors that do not obscure "
            "meaning, and no tool can judge it for you — use your own language understanding."
        )
    elif tool_context:
        system_parts.append(f"\nLinguistic analysis findings (for reference):\n{tool_context}")

    system_parts.append(
        "\nStructure your response as follows:\n"
        "<score>X.X</score>  — IELTS band score for this component (0–9, 0.5 increments)\n"
        "<confidence>high|medium|low</confidence>  — your confidence in the score\n"
        "\nThen provide your detailed feedback."
    )

    return [
        Message(role="system", content="\n\n".join(system_parts)),
        Message(role="user", content=f"Question:\n{question}\n\nEssay:\n{essay}"),
    ]


# ---------------------------------------------------------------------------
# Tool execution loop (used when strategy.use_tool_calling=True)
# ---------------------------------------------------------------------------

_EMPTY_CONTENT_RETRIES = 3


async def _acomplete_with_retry(provider, messages, model, enable_thinking, tools=None):
    """Call acomplete with retry on empty-content responses (model stops without producing output)."""
    for attempt in range(_EMPTY_CONTENT_RETRIES):
        try:
            return await provider.acomplete(
                messages, model=model, enable_thinking=enable_thinking, tools=tools
            )
        except RuntimeError as exc:
            if "empty content" in str(exc) and attempt < _EMPTY_CONTENT_RETRIES - 1:
                wait = 2 ** attempt
                logger.warning("Empty content on attempt %d, retrying in %ds", attempt + 1, wait)
                await asyncio.sleep(wait)
            else:
                raise


async def _run_tool_loop(
    messages: list[Message],
    tools: list[dict],
    provider: LLMProvider,
    model: str,
    enable_thinking: bool,
) -> str:
    from edsmith.examiner.tool_defs import execute_tool

    for _ in range(_MAX_TOOL_ROUNDS):
        response = await _acomplete_with_retry(
            provider, messages, model, enable_thinking, tools=tools
        )

        if not response.tool_calls:
            return response.content

        # Append assistant turn with the tool calls
        messages.append(Message(role="assistant", content=response.content or None, tool_calls=response.tool_calls))

        # Execute each tool and append results
        for tc in response.tool_calls:
            result = await asyncio.to_thread(execute_tool, tc.name, tc.arguments)
            logger.debug("tool_call name=%s result_len=%d", tc.name, len(result))
            messages.append(Message(role="tool", content=result, tool_call_id=tc.id))

    # Max rounds reached — final call without tools
    return (await _acomplete_with_retry(provider, messages, model, enable_thinking)).content


# ---------------------------------------------------------------------------
# Per-component generation
# ---------------------------------------------------------------------------

async def _generate_component(
    question: str,
    essay: str,
    component: str,
    policy: PromptPolicy,
    strategy: StrategyGuidance,
    tool_context: str,
    provider: LLMProvider,
    model: str,
    band: float | None = None,
    enable_thinking: bool = False,
) -> ComponentFeedback:
    messages = _build_messages(question, essay, component, policy, strategy, tool_context, band=band)

    if strategy.use_tool_calling:
        from edsmith.examiner.tool_defs import get_tool_definitions
        tools = get_tool_definitions(strategy)
        text = await _run_tool_loop(messages, tools, provider, model, enable_thinking)
    else:
        response = await provider.acomplete(messages, model=model, enable_thinking=enable_thinking)
        text = response.content

    return ComponentFeedback(
        component=component,
        feedback=text,
        score=_extract_score(text),
        tag=_extract_tag(text, "confidence"),
    )


# ---------------------------------------------------------------------------
# Score calibration (post-gather reflection pass)
# ---------------------------------------------------------------------------

_CALIBRATION_SYSTEM = """\
You are a senior IELTS examiner reviewing a set of component scores for consistency.
You are given four component scores and brief feedback excerpts, plus the verified
overall band for the essay. The four component scores must average to the overall band.

Identify which component score(s) should be adjusted (changing by 0.5–1.0 band steps
is normal; avoid large swings). For each adjusted component, provide a one-sentence
note explaining why the score was changed.

Respond ONLY with a JSON object inside <calibration> tags, nothing else:
<calibration>
{
  "adjustments": {
    "<component_key>": {"score": <new_score>, "note": "<one sentence>"}
  }
}
</calibration>
Only include components that need adjustment. If none are needed, return {"adjustments": {}}.
Valid component keys: task_response, coherence, lexical, grammar.
Scores must be 0–9 in 0.5 increments.
"""


async def _reflect_and_calibrate(
    feedbacks: dict[str, ComponentFeedback],
    band: float,
    provider: LLMProvider,
    model: str,
    enable_thinking: bool,
) -> dict[str, ComponentFeedback]:
    import json as _json

    scores = {k: fb.score for k, fb in feedbacks.items() if fb.score is not None}
    if len(scores) < 4:
        return feedbacks

    avg = sum(scores.values()) / 4
    if abs(avg - band) <= 0.25:
        return feedbacks

    logger.debug("calibration triggered avg=%.2f target=%s delta=%.2f", avg, band, avg - band)

    summaries = []
    for comp, fb in feedbacks.items():
        heading = COMPONENT_HEADINGS[comp]
        excerpt = (fb.feedback or "")[:300].strip().replace("\n", " ")
        summaries.append(f"{heading} (key={comp}): score={fb.score}\n  Excerpt: {excerpt}")

    user_content = (
        f"Verified overall band: {band}\n"
        f"Current component scores: {scores}\n"
        f"Current average: {avg:.2f}  |  Target: {band}  |  Delta: {avg - band:+.2f}\n\n"
        + "\n\n".join(summaries)
    )

    messages = [
        Message(role="system", content=_CALIBRATION_SYSTEM),
        Message(role="user", content=user_content),
    ]

    try:
        response = await provider.acomplete(messages, model=model, enable_thinking=enable_thinking)
        m = re.search(r"<calibration>(.*?)</calibration>", response.content, re.DOTALL | re.IGNORECASE)
        if not m:
            return feedbacks
        data = _json.loads(m.group(1))
        adjustments = data.get("adjustments", {})
    except Exception as exc:
        logger.warning("calibration failed: %s", exc)
        return feedbacks

    result = dict(feedbacks)
    for comp, adj in adjustments.items():
        if comp not in result:
            continue
        try:
            new_score = float(adj["score"])
            if not (0.0 <= new_score <= 9.0):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        note = adj.get("note", "")
        old_fb = result[comp]
        logger.debug("calibration %s: %.1f → %.1f  %s", comp, old_fb.score, new_score, note)
        delta = round(new_score - (old_fb.score or 0.0), 2)
        result[comp] = ComponentFeedback(
            component=comp,
            feedback=old_fb.feedback + f"\n\n[Score calibrated from {old_fb.score} to {new_score}: {note}]",
            score=new_score,
            tag=old_fb.tag,
            calibration_delta=delta,
        )

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_feedback(
    question: str,
    essay: str,
    policies: dict[str, PromptPolicy],
    strategy: StrategyGuidance,
    provider: LLMProvider,
    model_config: ModelConfig,
    band: float | None = None,
) -> dict[str, ComponentFeedback]:
    """Generate per-component feedback for a single essay.

    When strategy.use_tool_calling is False, linguistic tools are collected once
    (in a thread) before launching concurrent per-component LLM calls.
    When True, tools are passed as API function definitions and the LLM calls
    them dynamically during its reasoning.
    """
    if strategy.use_tool_calling:
        tool_context = ""
    else:
        tool_context = await asyncio.to_thread(_collect_tool_context_sync, essay, strategy)

    tasks = [
        _generate_component(
            question=question,
            essay=essay,
            component=component,
            policy=policies.get(component, PromptPolicy()),
            strategy=strategy,
            tool_context=tool_context,
            provider=provider,
            model=model_config.generator,
            band=band,
            enable_thinking=model_config.enable_thinking,
        )
        for component in COMPONENT_HEADINGS
    ]

    results = await asyncio.gather(*tasks)
    feedbacks = {fb.component: fb for fb in results}

    if band is not None:
        feedbacks = await _reflect_and_calibrate(
            feedbacks=feedbacks,
            band=band,
            provider=provider,
            model=model_config.generator,
            enable_thinking=model_config.enable_thinking,
        )

    return feedbacks
