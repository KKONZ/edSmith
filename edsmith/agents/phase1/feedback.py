from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from edsmith.config.session import PromptPolicy
from edsmith.data.parser import COMPONENT_HEADINGS, ComponentEval
from edsmith.providers.base import LLMProvider, Message


# ------------------------------------------------------------------
# IELTS band descriptors — condensed for prompting
# ------------------------------------------------------------------

_RUBRICS: dict[str, str] = {
    "task_response": (
        "Addresses all parts of the task with a fully developed position. "
        "Ideas are extended, supported, and relevant. "
        "Band 9: full address; Band 7: covers task; Band 5: partial; Band 3: barely addresses."
    ),
    "coherence": (
        "Information is logically sequenced and cohesive devices flow naturally. "
        "Paragraphing is appropriate and supports the argument. "
        "Band 9: seamless; Band 7: effective; Band 5: some lapses; Band 3: limited control."
    ),
    "lexical": (
        "Vocabulary is used with range, flexibility, and sophistication. "
        "Awareness of style, collocation, and word formation. "
        "Band 9: wide range; Band 7: sufficient; Band 5: limited; Band 3: very limited."
    ),
    "grammar": (
        "Wide range of structures used with full flexibility and accuracy. "
        "Complex sentences are attempted; errors are rare and do not impede communication. "
        "Band 9: near-perfect; Band 7: mostly accurate; Band 5: some errors; Band 3: frequent errors."
    ),
}

_SPECIFICITY_INSTRUCTIONS: dict[int, str] = {
    1: "Keep feedback brief (2–3 sentences, high-level only).",
    2: "Provide concise feedback with one or two specific observations.",
    3: "Provide moderate-length feedback with clear specific examples drawn from the essay.",
    4: "Provide detailed feedback with multiple quotes and examples from the essay.",
    5: (
        "Provide comprehensive, highly specific feedback: cite exact phrases, "
        "explain why each is strong or weak, and suggest targeted improvements."
    ),
}

_SYSTEM_TEMPLATE = """\
You are an expert IELTS examiner specialising in {component_name}.

## Component Rubric
{rubric}

## Feedback Guidelines
{guidelines}

## Output Format
<feedback>
Your feedback text for the student.
</feedback>
<score>NUMBER</score>

Score must be 0–9 in 0.5 increments (e.g. 5.0, 6.5, 7.0)."""


# ------------------------------------------------------------------
# Output dataclass
# ------------------------------------------------------------------

@dataclass
class ComponentFeedback:
    component: str
    text: str            # feedback text — becomes training target
    score: float | None  # predicted component score (0–9 in 0.5 steps)
    input_tokens: int
    output_tokens: int

    def to_component_eval(self) -> ComponentEval:
        return ComponentEval(text=self.text, score=self.score)


# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------

class FeedbackAgent:
    """Phase 1 feedback generation agent.

    Evaluates a single (question, essay, component) tuple against the
    IELTS rubric. Output is feedback text and a component score; CoT
    is not injected here — that happens during training data preprocessing.
    """

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> None:
        self._provider = provider
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Single component
    # ------------------------------------------------------------------

    def generate(
        self,
        question: str,
        essay: str,
        component: str,
        policy: PromptPolicy,
    ) -> ComponentFeedback:
        _validate_component(component)
        messages = _build_messages(question, essay, component, policy)
        resp = self._provider.complete(messages, self._model, self._temperature, self._max_tokens)
        return _parse_response(component, resp.content, resp.input_tokens, resp.output_tokens)

    async def agenerate(
        self,
        question: str,
        essay: str,
        component: str,
        policy: PromptPolicy,
    ) -> ComponentFeedback:
        _validate_component(component)
        messages = _build_messages(question, essay, component, policy)
        resp = await self._provider.acomplete(messages, self._model, self._temperature, self._max_tokens)
        return _parse_response(component, resp.content, resp.input_tokens, resp.output_tokens)

    # ------------------------------------------------------------------
    # All four components
    # ------------------------------------------------------------------

    def generate_all(
        self,
        question: str,
        essay: str,
        policies: dict[str, PromptPolicy],
    ) -> dict[str, ComponentFeedback]:
        return {
            component: self.generate(question, essay, component, policies[component])
            for component in COMPONENT_HEADINGS
        }

    async def agenerate_all(
        self,
        question: str,
        essay: str,
        policies: dict[str, PromptPolicy],
    ) -> dict[str, ComponentFeedback]:
        tasks = [
            self.agenerate(question, essay, component, policies[component])
            for component in COMPONENT_HEADINGS
        ]
        results = await asyncio.gather(*tasks)
        return dict(zip(COMPONENT_HEADINGS, results))


# ------------------------------------------------------------------
# Prompt construction
# ------------------------------------------------------------------

def _build_messages(
    question: str,
    essay: str,
    component: str,
    policy: PromptPolicy,
) -> list[Message]:
    component_name = COMPONENT_HEADINGS[component]

    guideline_lines = [_SPECIFICITY_INSTRUCTIONS[policy.specificity]]
    guideline_lines.append(
        "You must cite specific phrases or sentences from the essay as evidence."
        if policy.evidence_required
        else "General observations are acceptable; specific quotes are optional."
    )
    if policy.additional_instructions:
        guideline_lines.append(policy.additional_instructions)

    system = _SYSTEM_TEMPLATE.format(
        component_name=component_name,
        rubric=_RUBRICS[component],
        guidelines="\n".join(guideline_lines),
    )
    user = (
        f"## Question\n{question}\n\n"
        f"## Essay\n{essay}\n\n"
        f"Evaluate this essay on **{component_name}**."
    )
    return [Message(role="system", content=system), Message(role="user", content=user)]


# ------------------------------------------------------------------
# Response parsing
# ------------------------------------------------------------------

def _parse_response(
    component: str,
    content: str,
    input_tokens: int,
    output_tokens: int,
) -> ComponentFeedback:
    return ComponentFeedback(
        component=component,
        text=_extract_tag(content, "feedback"),
        score=_extract_score(content),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _extract_tag(content: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", content, re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_score(content: str) -> float | None:
    m = re.search(r"<score>\s*(\d+(?:\.\d+)?)\s*</score>", content)
    if not m:
        import sys
        print(f"  [feedback] no <score> tag in response (first 120 chars): {content[:120]!r}", file=sys.stderr)
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    if not (0.0 <= val <= 9.0):
        return None
    return round(val * 2) / 2  # snap to nearest 0.5 IELTS band step


def _validate_component(component: str) -> None:
    if component not in COMPONENT_HEADINGS:
        raise ValueError(
            f"Unknown component {component!r}. Must be one of {list(COMPONENT_HEADINGS)}."
        )
