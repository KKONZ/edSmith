from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field

from edsmith.config.session import CouncilConfig, PromptPolicy
from edsmith.data.parser import COMPONENT_HEADINGS, ComponentEval
from edsmith.memory.semantic import SemanticExample, SemanticMemory
from edsmith.providers.base import LLMProvider, Message


# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------

_GENERATOR_SYSTEM = """\
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

_CRITIC_SYSTEM = """\
You are a senior IELTS examiner peer-reviewing feedback written by a colleague for \
{component_name}.

## Component Rubric
{rubric}

Evaluate whether the proposed feedback and score are well-calibrated. Consider:
- Is the score accurate against the rubric band descriptors?
- Is the feedback specific, fair, and evidence-based?
- Are important strengths or weaknesses overlooked?
- Is anything misleading or incorrect?

## Output Format
<critique>
Your critical assessment of the feedback and score.
</critique>"""

_CHAIR_SYSTEM = """\
You are the chief IELTS examiner chairing a panel for {component_name}.

## Component Rubric
{rubric}
{memory_block}
You have received a deliberation between a Generator and a Critic. Synthesise the \
discussion into a final, authoritative feedback and score. You may agree with either \
party or strike a balance — justify your decision in the feedback.

## Output Format
<feedback>
The final feedback text for the student.
</feedback>
<score>NUMBER</score>

Score must be 0–9 in 0.5 increments (e.g. 5.0, 6.5, 7.0)."""

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


# ------------------------------------------------------------------
# Output dataclasses
# ------------------------------------------------------------------

@dataclass
class Round:
    draft: str
    critique: str


@dataclass
class ComponentFeedback:
    component: str
    text: str
    score: float | None
    rounds: list[Round] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0

    def to_component_eval(self) -> ComponentEval:
        return ComponentEval(text=self.text, score=self.score)


# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------

class CouncilAgent:
    """Phase 1 LLM-Council feedback agent.

    Runs Generator → Critic for ``critic_rounds`` cycles, then the Chair
    synthesises a final ComponentFeedback.  When ``chair_memory_injection``
    is enabled, the Chair receives semantically similar examples from the
    training collection to help calibrate its final score.

    All three roles use the provider with their own model strings sourced
    from ``ModelConfig``.
    """

    def __init__(
        self,
        provider: LLMProvider,
        generator_model: str,
        critic_model: str,
        chair_model: str,
        critic_rounds: int = 1,
        chair_memory: SemanticMemory | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> None:
        self._provider = provider
        self._generator_model = generator_model
        self._critic_model = critic_model
        self._chair_model = chair_model
        self._critic_rounds = max(1, critic_rounds)
        self._chair_memory = chair_memory
        self._temperature = temperature
        self._max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Public API — single component
    # ------------------------------------------------------------------

    def generate(
        self,
        question: str,
        essay: str,
        component: str,
        policy: PromptPolicy,
        k: int = 4,
        essay_id: str | None = None,
    ) -> ComponentFeedback:
        _validate_component(component)
        return self._deliberate(question, essay, component, policy, k, essay_id)

    async def agenerate(
        self,
        question: str,
        essay: str,
        component: str,
        policy: PromptPolicy,
        k: int = 4,
        essay_id: str | None = None,
    ) -> ComponentFeedback:
        _validate_component(component)
        return await self._adeliberate(question, essay, component, policy, k, essay_id)

    # ------------------------------------------------------------------
    # Public API — all four components
    # ------------------------------------------------------------------

    def generate_all(
        self,
        question: str,
        essay: str,
        policies: dict[str, PromptPolicy],
        k: int = 4,
        essay_id: str | None = None,
    ) -> dict[str, ComponentFeedback]:
        return {
            component: self.generate(question, essay, component, policies[component], k, essay_id)
            for component in COMPONENT_HEADINGS
        }

    async def agenerate_all(
        self,
        question: str,
        essay: str,
        policies: dict[str, PromptPolicy],
        k: int = 4,
        essay_id: str | None = None,
    ) -> dict[str, ComponentFeedback]:
        tasks = [
            self.agenerate(question, essay, component, policies[component], k, essay_id)
            for component in COMPONENT_HEADINGS
        ]
        results = await asyncio.gather(*tasks)
        return dict(zip(COMPONENT_HEADINGS, results))

    # ------------------------------------------------------------------
    # Deliberation — sync
    # ------------------------------------------------------------------

    def _deliberate(
        self,
        question: str,
        essay: str,
        component: str,
        policy: PromptPolicy,
        k: int,
        essay_id: str | None,
    ) -> ComponentFeedback:
        total_in = total_out = 0
        component_name = COMPONENT_HEADINGS[component]
        rubric = _RUBRICS[component]
        guidelines = _guidelines(policy)

        # Generator conversation — accumulates revisions across rounds
        gen_msgs: list[Message] = [
            Message(role="system", content=_GENERATOR_SYSTEM.format(
                component_name=component_name, rubric=rubric, guidelines=guidelines,
            )),
            Message(role="user", content=_essay_prompt(question, essay, component_name)),
        ]

        # Critic conversation — accumulates critiques across rounds
        crit_msgs: list[Message] = [
            Message(role="system", content=_CRITIC_SYSTEM.format(
                component_name=component_name, rubric=rubric,
            )),
        ]

        rounds: list[Round] = []

        for round_i in range(self._critic_rounds):
            # Generator drafts (or revises)
            resp = self._provider.complete(gen_msgs, self._generator_model, self._temperature, self._max_tokens)
            total_in += resp.input_tokens; total_out += resp.output_tokens
            draft = resp.content
            gen_msgs.append(Message(role="assistant", content=draft))

            # Critic reviews
            if round_i == 0:
                crit_msgs.append(Message(role="user", content=_critic_prompt(question, essay, component_name, draft)))
            else:
                crit_msgs.append(Message(role="user", content=f"The generator revised their feedback:\n\n{draft}\n\nPlease critique the revision."))

            resp = self._provider.complete(crit_msgs, self._critic_model, self._temperature, self._max_tokens)
            total_in += resp.input_tokens; total_out += resp.output_tokens
            critique = resp.content
            crit_msgs.append(Message(role="assistant", content=critique))

            rounds.append(Round(draft=draft, critique=critique))

            # Prime generator for next round
            if round_i < self._critic_rounds - 1:
                gen_msgs.append(Message(role="user", content=f"The critic said:\n\n{critique}\n\nPlease revise your feedback."))

        # Chair synthesises
        memory_examples = self._fetch_memory(essay, component, k, essay_id)
        chair_msgs = [
            Message(role="system", content=_CHAIR_SYSTEM.format(
                component_name=component_name,
                rubric=rubric,
                memory_block=_memory_block(memory_examples, component_name),
            )),
            Message(role="user", content=_chair_prompt(question, essay, component_name, rounds)),
        ]
        resp = self._provider.complete(chair_msgs, self._chair_model, self._temperature, self._max_tokens)
        total_in += resp.input_tokens; total_out += resp.output_tokens

        return _parse_response(component, resp.content, rounds, total_in, total_out)

    # ------------------------------------------------------------------
    # Deliberation — async
    # ------------------------------------------------------------------

    async def _adeliberate(
        self,
        question: str,
        essay: str,
        component: str,
        policy: PromptPolicy,
        k: int,
        essay_id: str | None,
    ) -> ComponentFeedback:
        total_in = total_out = 0
        component_name = COMPONENT_HEADINGS[component]
        rubric = _RUBRICS[component]
        guidelines = _guidelines(policy)

        gen_msgs: list[Message] = [
            Message(role="system", content=_GENERATOR_SYSTEM.format(
                component_name=component_name, rubric=rubric, guidelines=guidelines,
            )),
            Message(role="user", content=_essay_prompt(question, essay, component_name)),
        ]
        crit_msgs: list[Message] = [
            Message(role="system", content=_CRITIC_SYSTEM.format(
                component_name=component_name, rubric=rubric,
            )),
        ]
        rounds: list[Round] = []

        for round_i in range(self._critic_rounds):
            resp = await self._provider.acomplete(gen_msgs, self._generator_model, self._temperature, self._max_tokens)
            total_in += resp.input_tokens; total_out += resp.output_tokens
            draft = resp.content
            gen_msgs.append(Message(role="assistant", content=draft))

            if round_i == 0:
                crit_msgs.append(Message(role="user", content=_critic_prompt(question, essay, component_name, draft)))
            else:
                crit_msgs.append(Message(role="user", content=f"The generator revised their feedback:\n\n{draft}\n\nPlease critique the revision."))

            resp = await self._provider.acomplete(crit_msgs, self._critic_model, self._temperature, self._max_tokens)
            total_in += resp.input_tokens; total_out += resp.output_tokens
            critique = resp.content
            crit_msgs.append(Message(role="assistant", content=critique))

            rounds.append(Round(draft=draft, critique=critique))

            if round_i < self._critic_rounds - 1:
                gen_msgs.append(Message(role="user", content=f"The critic said:\n\n{critique}\n\nPlease revise your feedback."))

        memory_examples = self._fetch_memory(essay, component, k, essay_id)
        chair_msgs = [
            Message(role="system", content=_CHAIR_SYSTEM.format(
                component_name=component_name,
                rubric=rubric,
                memory_block=_memory_block(memory_examples, component_name),
            )),
            Message(role="user", content=_chair_prompt(question, essay, component_name, rounds)),
        ]
        resp = await self._provider.acomplete(chair_msgs, self._chair_model, self._temperature, self._max_tokens)
        total_in += resp.input_tokens; total_out += resp.output_tokens

        return _parse_response(component, resp.content, rounds, total_in, total_out)

    # ------------------------------------------------------------------
    # Memory helper
    # ------------------------------------------------------------------

    def _fetch_memory(
        self,
        essay: str,
        component: str,
        k: int,
        essay_id: str | None,
    ) -> list[SemanticExample]:
        if self._chair_memory is None:
            return []
        return self._chair_memory.retrieve_train(essay, component, k=k, exclude_id=essay_id)


# ------------------------------------------------------------------
# Prompt builders
# ------------------------------------------------------------------

def _guidelines(policy: PromptPolicy) -> str:
    lines = [_SPECIFICITY_INSTRUCTIONS[policy.specificity]]
    lines.append(
        "You must cite specific phrases or sentences from the essay as evidence."
        if policy.evidence_required
        else "General observations are acceptable; specific quotes are optional."
    )
    if policy.additional_instructions:
        lines.append(policy.additional_instructions)
    return "\n".join(lines)


def _essay_prompt(question: str, essay: str, component_name: str) -> str:
    return (
        f"## Question\n{question}\n\n"
        f"## Essay\n{essay}\n\n"
        f"Evaluate this essay on **{component_name}**."
    )


def _critic_prompt(question: str, essay: str, component_name: str, draft: str) -> str:
    return (
        f"## Question\n{question}\n\n"
        f"## Essay\n{essay}\n\n"
        f"## Generator's Feedback ({component_name})\n{draft}\n\n"
        "Please critique this feedback and score."
    )


def _chair_prompt(question: str, essay: str, component_name: str, rounds: list[Round]) -> str:
    parts = [f"## Question\n{question}\n\n## Essay\n{essay}\n\n## Deliberation"]
    for i, r in enumerate(rounds, 1):
        parts.append(f"\n### Round {i}\n**Generator:**\n{r.draft}\n\n**Critic:**\n{r.critique}")
    parts.append(f"\nSynthesise the deliberation above into a final {component_name} feedback and score.")
    return "\n".join(parts)


def _memory_block(examples: list[SemanticExample], component_name: str) -> str:
    if not examples:
        return ""
    lines = ["\n## Reference Examples from Memory"]
    for i, ex in enumerate(examples, 1):
        score_str = str(ex.score) if ex.score is not None else "N/A"
        lines.append(
            f"\n[{i}] Overall band {ex.band} | {component_name}: {score_str}\n"
            f"Essay: {ex.essay}\n"
            f"Feedback: {ex.feedback_text}"
        )
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Response parsing
# ------------------------------------------------------------------

def _parse_response(
    component: str,
    content: str,
    rounds: list[Round],
    input_tokens: int,
    output_tokens: int,
) -> ComponentFeedback:
    return ComponentFeedback(
        component=component,
        text=_extract_tag(content, "feedback"),
        score=_extract_score(content),
        rounds=rounds,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _extract_tag(content: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", content, re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_score(content: str) -> float | None:
    m = re.search(r"<score>\s*(\d+(?:\.\d+)?)\s*</score>", content)
    if not m:
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    if not (0.0 <= val <= 9.0):
        return None
    return round(val * 2) / 2


def _validate_component(component: str) -> None:
    if component not in COMPONENT_HEADINGS:
        raise ValueError(
            f"Unknown component {component!r}. Must be one of {list(COMPONENT_HEADINGS)}."
        )
