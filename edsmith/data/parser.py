from __future__ import annotations

import re
from dataclasses import dataclass, field


COMPONENT_HEADINGS: dict[str, str] = {
    "task_response": "Task Achievement",
    "coherence": "Coherence and Cohesion",
    "lexical": "Lexical Resource",
    "grammar": "Grammatical Range and Accuracy",
}

_EXTRA_HEADINGS = [
    "Overall Band Score",
    "Feedback and Additional Comments",
    "Suggestions for Enhancement",
]

_VALID_SCORE_RANGE = (0.0, 9.0)


@dataclass
class ComponentEval:
    text: str = ""
    score: float | None = None


@dataclass
class ParsedEvaluation:
    components: dict[str, ComponentEval] = field(default_factory=dict)
    overall_feedback: str = ""
    suggestions: str = ""


def _extract_score_from_heading(heading: str) -> float | None:
    # **Task Achievement: [7]** or [7.5]
    m = re.search(r"\[(\d+(?:\.\d+)?)\]", heading)
    if m:
        return _validated_score(m.group(1))

    # **Task Achievement (7):** — parenthesised score
    m = re.search(r"\((\d+(?:\.\d+)?)\)", heading)
    if m:
        return _validated_score(m.group(1))

    # **: 7.0  /  : 7.0  /  ** 7.0  at end of heading or before a dash separator
    # e.g. "**Task Achievement:** 5.0 - The candidate..."
    m = re.search(r"[:\*]\*?\s*(\d+(?:\.\d+)?)\s*(?:\*|\n|$|(?=\s*-))", heading)
    if m:
        return _validated_score(m.group(1))

    return None


def _validated_score(raw: str) -> float | None:
    try:
        val = float(raw)
    except ValueError:
        return None
    lo, hi = _VALID_SCORE_RANGE
    return val if lo <= val <= hi else None


def _clean_text(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line = re.sub(r"^[-*+•]\s*", "", line.lstrip(), count=1)
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _build_heading_pattern() -> re.Pattern[str]:
    all_labels = list(COMPONENT_HEADINGS.values()) + _EXTRA_HEADINGS
    parts = [rf"[#\*\s\d\.]*{re.escape(label)}[^\n:]*:?[^\n]*" for label in all_labels]
    return re.compile(r"^(" + "|".join(parts) + r")[ \t]*\n?", re.IGNORECASE | re.MULTILINE)


_HEADING_PATTERN = _build_heading_pattern()


def _resolve_section_key(heading_text: str) -> str | None:
    for key, label in COMPONENT_HEADINGS.items():
        if re.search(re.escape(label), heading_text, re.IGNORECASE):
            return key
    if re.search(r"Overall Band Score|Feedback and Additional Comments", heading_text, re.IGNORECASE):
        return "overall_feedback"
    if re.search(r"Suggestions for Enhancement", heading_text, re.IGNORECASE):
        return "suggestions"
    return None


def parse_evaluation(eval_text: str) -> ParsedEvaluation:
    """Extract per-Component text and scores from a raw evaluation string."""
    seen_keys: set[str] = set()
    heading_spans: list[tuple[int, int, str, str]] = []

    for match in _HEADING_PATTERN.finditer(eval_text):
        heading_text = match.group(0)
        if re.match(r"^\s*[-*+•]\s+\**", heading_text):
            continue  # bullet item, not a section heading
        key = _resolve_section_key(heading_text)
        if key and key not in seen_keys:
            heading_spans.append((match.start(), match.end(), heading_text, key))
            seen_keys.add(key)

    heading_spans.sort(key=lambda x: x[0])

    raw: dict[str, tuple[str, str]] = {}
    for i, (_, end, heading_text, key) in enumerate(heading_spans):
        content_end = heading_spans[i + 1][0] if i + 1 < len(heading_spans) else len(eval_text)
        raw[key] = (heading_text, eval_text[end:content_end].strip())

    result = ParsedEvaluation()
    for component_key in COMPONENT_HEADINGS:
        if component_key in raw:
            heading_text, content = raw[component_key]
            result.components[component_key] = ComponentEval(
                text=_clean_text(content),
                score=_extract_score_from_heading(heading_text),
            )
        else:
            result.components[component_key] = ComponentEval()

    result.overall_feedback = _clean_text(raw.get("overall_feedback", ("", ""))[1])
    result.suggestions = _clean_text(raw.get("suggestions", ("", ""))[1])
    return result
