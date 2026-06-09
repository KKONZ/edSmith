import pytest

from edsmith.data.parser import (
    _extract_score_from_heading,
    _validated_score,
    parse_evaluation,
)


# ---------------------------------------------------------------------------
# Sample evaluation texts
# ---------------------------------------------------------------------------

_FULL_EVAL = """\
## Task Achievement: [7]
The essay addresses all parts of the task. The writer presents a clear position.

## Coherence and Cohesion (6.5):
Structure is generally clear but some cohesive devices are overused.

## Lexical Resource: 7.0
A good range of vocabulary with only minor inaccuracies.

## Grammatical Range and Accuracy: 5.5
Frequent minor errors are present but communication is not severely impeded.

## Overall Band Score
The overall performance is solid.

## Suggestions for Enhancement
Focus on reducing grammatical errors.
"""

_PARTIAL_EVAL = """\
## Task Achievement: [7]
Good addressing of the task.

## Overall Band Score
Strong overall.
"""

_EMPTY_EVAL = ""


# ---------------------------------------------------------------------------
# parse_evaluation
# ---------------------------------------------------------------------------

class TestParseEvaluation:
    def test_extracts_all_four_components(self):
        result = parse_evaluation(_FULL_EVAL)
        assert set(result.components.keys()) == {"task_response", "coherence", "lexical", "grammar"}

    def test_scores_extracted_correctly(self):
        result = parse_evaluation(_FULL_EVAL)
        assert result.components["task_response"].score == 7.0
        assert result.components["coherence"].score == 6.5
        assert result.components["lexical"].score == 7.0
        assert result.components["grammar"].score == 5.5

    def test_component_text_non_empty(self):
        result = parse_evaluation(_FULL_EVAL)
        for key, comp in result.components.items():
            assert len(comp.text) > 0, f"Empty text for {key}"

    def test_overall_feedback_captured(self):
        result = parse_evaluation(_FULL_EVAL)
        assert "solid" in result.overall_feedback.lower()

    def test_suggestions_captured(self):
        result = parse_evaluation(_FULL_EVAL)
        assert "grammatical" in result.suggestions.lower()

    def test_missing_components_have_empty_eval(self):
        result = parse_evaluation(_PARTIAL_EVAL)
        # coherence, lexical, grammar not present → empty ComponentEval
        assert result.components["coherence"].score is None
        assert result.components["coherence"].text == ""

    def test_empty_string(self):
        result = parse_evaluation(_EMPTY_EVAL)
        for comp in result.components.values():
            assert comp.score is None
            assert comp.text == ""

    def test_duplicate_heading_first_wins(self):
        # If a component appears twice, the first occurrence wins (seen_keys guard)
        text = """\
## Task Achievement: [7]
First section text.

## Task Achievement: [5]
Second section text.
"""
        result = parse_evaluation(text)
        assert result.components["task_response"].score == 7.0


# ---------------------------------------------------------------------------
# _extract_score_from_heading
# ---------------------------------------------------------------------------

class TestExtractScoreFromHeading:
    def test_bracket_integer(self):
        assert _extract_score_from_heading("## Task Achievement: [7]") == 7.0

    def test_bracket_decimal(self):
        assert _extract_score_from_heading("## Task Achievement: [6.5]") == 6.5

    def test_parenthesis_format(self):
        assert _extract_score_from_heading("## Coherence and Cohesion (6.5):") == 6.5

    def test_colon_decimal(self):
        assert _extract_score_from_heading("## Lexical Resource: 7.0") == 7.0

    def test_out_of_range_returns_none(self):
        assert _extract_score_from_heading("## Task Achievement: [10]") is None

    def test_no_score_returns_none(self):
        assert _extract_score_from_heading("## Task Achievement") is None


# ---------------------------------------------------------------------------
# _validated_score
# ---------------------------------------------------------------------------

class TestValidatedScore:
    def test_valid_integer(self):
        assert _validated_score("7") == 7.0

    def test_valid_decimal(self):
        assert _validated_score("6.5") == 6.5

    def test_boundary_zero(self):
        assert _validated_score("0") == 0.0

    def test_boundary_nine(self):
        assert _validated_score("9") == 9.0

    def test_above_range(self):
        assert _validated_score("10") is None

    def test_below_range(self):
        assert _validated_score("-1") is None

    def test_invalid_string(self):
        assert _validated_score("abc") is None
