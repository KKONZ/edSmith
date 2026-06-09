import asyncio

import pytest

from edsmith.agents.phase1.feedback import (
    FeedbackAgent,
    _build_messages,
    _extract_score,
    _extract_tag,
    _validate_component,
)
from edsmith.config.session import PromptPolicy

_DEFAULT_POLICY = PromptPolicy(specificity=2, evidence_required=True)


# ---------------------------------------------------------------------------
# _extract_score
# ---------------------------------------------------------------------------

class TestExtractScore:
    def test_integer_score(self):
        assert _extract_score("<score>7</score>") == 7.0

    def test_decimal_score(self):
        assert _extract_score("<score>6.5</score>") == 6.5

    def test_snaps_to_nearest_half_up(self):
        # 6.3 → rounds to 6.5 (nearest 0.5)
        assert _extract_score("<score>6.3</score>") == 6.5

    def test_snaps_to_nearest_half_down(self):
        # 6.1 → rounds to 6.0
        assert _extract_score("<score>6.1</score>") == 6.0

    def test_boundary_zero(self):
        assert _extract_score("<score>0</score>") == 0.0

    def test_boundary_nine(self):
        assert _extract_score("<score>9</score>") == 9.0

    def test_out_of_range_none(self):
        assert _extract_score("<score>10</score>") is None

    def test_missing_tag_none(self):
        assert _extract_score("no score here") is None

    def test_whitespace_in_tag(self):
        assert _extract_score("<score> 7.0 </score>") == 7.0


# ---------------------------------------------------------------------------
# _extract_tag
# ---------------------------------------------------------------------------

class TestExtractTag:
    def test_present(self):
        assert _extract_tag("<feedback>Good essay.</feedback>", "feedback") == "Good essay."

    def test_absent_returns_empty(self):
        assert _extract_tag("no tag here", "feedback") == ""

    def test_multiline_content(self):
        content = "<feedback>\nLine 1\nLine 2\n</feedback>"
        assert _extract_tag(content, "feedback") == "Line 1\nLine 2"

    def test_score_tag(self):
        assert _extract_tag("<score>6.5</score>", "score") == "6.5"


# ---------------------------------------------------------------------------
# _validate_component
# ---------------------------------------------------------------------------

class TestValidateComponent:
    @pytest.mark.parametrize("component", ["task_response", "coherence", "lexical", "grammar"])
    def test_valid_components_do_not_raise(self, component):
        _validate_component(component)  # must not raise

    def test_invalid_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown component"):
            _validate_component("criteria")

    def test_wrong_case_raises(self):
        with pytest.raises(ValueError):
            _validate_component("Coherence")


# ---------------------------------------------------------------------------
# _build_messages
# ---------------------------------------------------------------------------

class TestBuildMessages:
    def test_returns_two_messages(self):
        msgs = _build_messages("Q", "Essay.", "grammar", _DEFAULT_POLICY)
        assert len(msgs) == 2

    def test_roles(self):
        msgs = _build_messages("Q", "Essay.", "grammar", _DEFAULT_POLICY)
        assert msgs[0].role == "system"
        assert msgs[1].role == "user"

    def test_system_contains_component_name(self):
        msgs = _build_messages("Q", "Essay.", "grammar", _DEFAULT_POLICY)
        assert "Grammatical Range and Accuracy" in msgs[0].content

    def test_user_contains_essay_and_question(self):
        msgs = _build_messages("Discuss climate change.", "My essay.", "task_response", _DEFAULT_POLICY)
        assert "Discuss climate change." in msgs[1].content
        assert "My essay." in msgs[1].content

    def test_specificity_5_includes_comprehensive(self):
        policy = PromptPolicy(specificity=5)
        msgs = _build_messages("Q", "E", "lexical", policy)
        assert "comprehensive" in msgs[0].content.lower()

    def test_evidence_required_true(self):
        policy = PromptPolicy(evidence_required=True)
        msgs = _build_messages("Q", "E", "coherence", policy)
        assert "evidence" in msgs[0].content.lower() or "cite" in msgs[0].content.lower()

    def test_evidence_required_false(self):
        policy = PromptPolicy(evidence_required=False)
        msgs = _build_messages("Q", "E", "coherence", policy)
        assert "optional" in msgs[0].content.lower()

    def test_additional_instructions_included(self):
        policy = PromptPolicy(additional_instructions="Focus on academic register.")
        msgs = _build_messages("Q", "E", "coherence", policy)
        assert "academic register" in msgs[0].content


# ---------------------------------------------------------------------------
# FeedbackAgent.generate (sync, with stub provider)
# ---------------------------------------------------------------------------

class TestFeedbackAgentGenerate:
    def test_returns_correct_component(self, stub_provider):
        stub_provider.set("<feedback>Clear.</feedback><score>7.0</score>")
        agent = FeedbackAgent(provider=stub_provider, model="stub")
        result = agent.generate("Q", "E", "task_response", _DEFAULT_POLICY)
        assert result.component == "task_response"

    def test_score_parsed(self, stub_provider):
        stub_provider.set("<feedback>Good.</feedback><score>6.5</score>")
        agent = FeedbackAgent(provider=stub_provider, model="stub")
        result = agent.generate("Q", "E", "coherence", _DEFAULT_POLICY)
        assert result.score == 6.5

    def test_text_parsed(self, stub_provider):
        stub_provider.set("<feedback>Strong vocabulary range.</feedback><score>7.0</score>")
        agent = FeedbackAgent(provider=stub_provider, model="stub")
        result = agent.generate("Q", "E", "lexical", _DEFAULT_POLICY)
        assert "Strong vocabulary range." in result.text

    def test_missing_score_is_none(self, stub_provider):
        stub_provider.set("<feedback>Some feedback.</feedback>")
        agent = FeedbackAgent(provider=stub_provider, model="stub")
        result = agent.generate("Q", "E", "grammar", _DEFAULT_POLICY)
        assert result.score is None

    def test_token_counts(self, stub_provider):
        stub_provider.set("<feedback>F.</feedback><score>5.0</score>")
        agent = FeedbackAgent(provider=stub_provider, model="stub")
        result = agent.generate("Q", "E", "grammar", _DEFAULT_POLICY)
        assert result.input_tokens == 10
        assert result.output_tokens == 20

    def test_invalid_component_raises(self, stub_provider):
        agent = FeedbackAgent(provider=stub_provider, model="stub")
        with pytest.raises(ValueError):
            agent.generate("Q", "E", "criteria", _DEFAULT_POLICY)

    def test_generate_all_returns_four_components(self, stub_provider):
        stub_provider.set("<feedback>F.</feedback><score>6.0</score>")
        agent = FeedbackAgent(provider=stub_provider, model="stub")
        policies = {c: _DEFAULT_POLICY for c in ("task_response", "coherence", "lexical", "grammar")}
        results = agent.generate_all("Q", "E", policies)
        assert set(results.keys()) == {"task_response", "coherence", "lexical", "grammar"}


# ---------------------------------------------------------------------------
# FeedbackAgent.agenerate (async, via asyncio.run)
# ---------------------------------------------------------------------------

class TestFeedbackAgentAsync:
    def test_agenerate_returns_score(self, stub_provider):
        stub_provider.set("<feedback>Clear structure.</feedback><score>6.5</score>")
        agent = FeedbackAgent(provider=stub_provider, model="stub")
        result = asyncio.run(agent.agenerate("Q", "E", "coherence", _DEFAULT_POLICY))
        assert result.score == 6.5
        assert result.component == "coherence"

    def test_agenerate_all_four_components(self, stub_provider):
        stub_provider.set("<feedback>F.</feedback><score>7.0</score>")
        agent = FeedbackAgent(provider=stub_provider, model="stub")
        policies = {c: _DEFAULT_POLICY for c in ("task_response", "coherence", "lexical", "grammar")}
        results = asyncio.run(agent.agenerate_all("Q", "E", policies))
        assert set(results.keys()) == {"task_response", "coherence", "lexical", "grammar"}
        for fb in results.values():
            assert fb.score == 7.0
