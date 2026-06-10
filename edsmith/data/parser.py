from __future__ import annotations

import re
from dataclasses import dataclass, field


COMPONENT_HEADINGS: dict[str, str] = {
    "task_response": "Task Achievement",
    "coherence": "Coherence and Cohesion",
    "lexical": "Lexical Resource",
    "grammar": "Grammatical Range and Accuracy",
}

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


# ------------------------------------------------------------------
# Score extraction
# ------------------------------------------------------------------

def _validated_score(raw: str) -> float | None:
    try:
        val = float(raw)
    except ValueError:
        return None
    lo, hi = _VALID_SCORE_RANGE
    return val if lo <= val <= hi else None


def _extract_score_from_heading(heading: str) -> float | None:
    # [7] or [7.5]
    m = re.search(r"\[(\d+(?:\.\d+)?)\]", heading)
    if m:
        return _validated_score(m.group(1))

    # (7) or (7.5)
    m = re.search(r"\((\d+(?:\.\d+)?)\)", heading)
    if m:
        return _validated_score(m.group(1))

    # **: 5.0  /  : 7.0  /  ** 7  — at end of heading or before a dash separator
    m = re.search(r"[:\*]\*?\s*(\d+(?:\.\d+)?)\s*(?:\*|\n|$|(?=\s*-))", heading)
    if m:
        return _validated_score(m.group(1))

    return None


# ------------------------------------------------------------------
# Section splitting (verbatim user logic)
# ------------------------------------------------------------------

def split_evaluation(eval_text):
    sections = {}

    heading_map = {
        "task_response": "Task Achievement",
        "coherence": "Coherence and Cohesion",
        "lexical": "Lexical Resource",
        "grammar": "Grammatical Range and Accuracy"
    }

    # Helper function to clean leading bullet points from each line
    def _clean_bullet_points(text):
        lines = text.splitlines()
        cleaned_lines = []
        for line in lines:
            lstripped_line = line.lstrip()
            # Remove leading bullet points (-, *, +, •) and any following spaces
            cleaned_line = re.sub(r"^[-*+•]\s*", "", lstripped_line, count=1)
            cleaned_lines.append(cleaned_line)

        return "\n".join(cleaned_lines).strip()

    # Allow optional markdown symbols like #, ##, **, * and optional numbers
    all_heading_regexes = []

    for criterion_key, heading_text in heading_map.items():
        all_heading_regexes.append(rf"^[#\*\s\d\.]*{re.escape(heading_text)}[^\n:]*:?[^\n]*(?:\n|$)")

    all_heading_regexes.append(rf"^[#\*\s\d\.]*Overall Band Score[^\n:]*:?[^\n]*(?:\n|$)")
    all_heading_regexes.append(rf"^[#\*\s\d\.]*Feedback and Additional Comments[^\n:]*:?[^\n]*(?:\n|$)")
    all_heading_regexes.append(rf"^[#\*\s\d\.]*Suggestions for Enhancement[^\n:]*:?[^\n]*(?:\n|$)")

    # Find all heading occurrences and their positions
    heading_matches = []
    combined_heading_pattern = "(" + "|".join(all_heading_regexes) + ")"

    # We only want to match the first time a heading appears as a main section
    # to avoid capturing summary bullets at the bottom.
    matched_keys = set()

    for match in re.finditer(combined_heading_pattern, eval_text, flags=re.IGNORECASE | re.MULTILINE):
        heading_text_full = match.group(0)

        section_name = ""
        for k, h_text in heading_map.items():
            if re.search(rf"(?i){re.escape(h_text)}", heading_text_full):
                section_name = k
                break

        if not section_name:
            if re.search(r"(?i)Overall Band Score", heading_text_full):
                section_name = "overall_band_score"
            elif re.search(r"(?i)Feedback and Additional Comments", heading_text_full):
                section_name = "overall_feedback"
            elif re.search(r"(?i)Suggestions for Enhancement", heading_text_full):
                section_name = "suggestions"

        # If we haven't seen this section yet (or if we want to allow overwriting, we don't check)
        # But usually the first match is the main section.
        if section_name and section_name not in matched_keys:
            # Except if the match is a bullet point, let's ignore it if we suspect it's a summary bullet
            if re.match(r"^\s*[-*+•]\s+\**", heading_text_full):
                continue # skip list items that look like headings

            heading_matches.append((match.start(), match.end(), heading_text_full, section_name))
            matched_keys.add(section_name)

    # Sort matches by their start position
    heading_matches.sort(key=lambda x: x[0])

    # Create sections based on these matches
    sections_raw_content = {}
    for i, (start, end, heading_text_full, section_name) in enumerate(heading_matches):
        content_start_pos = end # Content starts right after the full heading line

        # Find the end of the content for this section
        if i + 1 < len(heading_matches):
            content_end_pos = heading_matches[i+1][0] # Content ends at the start of the next heading
        else:
            content_end_pos = len(eval_text) # Last section, content goes to end of string

        content = eval_text[content_start_pos:content_end_pos].strip()
        sections_raw_content[section_name] = content

    # Populate final sections dictionary and apply cleaning
    for criterion_key in heading_map.keys():
        sections[criterion_key] = _clean_bullet_points(sections_raw_content.get(criterion_key, ""))

    sections["overall_feedback"] = _clean_bullet_points(sections_raw_content.get("overall_feedback", ""))
    sections["suggestions"] = _clean_bullet_points(sections_raw_content.get("suggestions", ""))

    return sections


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def parse_evaluation(eval_text: str) -> ParsedEvaluation:
    """Extract per-component text and scores from a raw evaluation string."""
    sections = split_evaluation(eval_text)

    # Re-scan for heading lines to extract embedded scores.
    result = ParsedEvaluation()
    for key, label in COMPONENT_HEADINGS.items():
        pat = re.compile(
            rf"^[#\*\s\d\.]*{re.escape(label)}[^\n:]*:?[^\n]*(?:\n|$)",
            re.IGNORECASE | re.MULTILINE,
        )
        m = pat.search(eval_text)
        score = _extract_score_from_heading(m.group(0)) if m else None
        result.components[key] = ComponentEval(text=sections[key], score=score)

    result.overall_feedback = sections["overall_feedback"]
    result.suggestions = sections["suggestions"]
    return result
