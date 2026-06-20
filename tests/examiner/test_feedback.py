import asyncio

import pandas as pd
import pytest

from edsmith.config.session import ModelConfig, PromptPolicy, StrategyGuidance
from edsmith.examiner.feedback import (
    ComponentFeedback,
    _extract_score,
    _extract_tag,
    generate_feedback,
)
from edsmith.session.state import SessionState, save_state

# ---------------------------------------------------------------------------
# _extract_score
# ---------------------------------------------------------------------------

class TestExtractScore:
    def test_xml_tag(self):
        assert _extract_score("<score>6.5</score>") == 6.5

    def test_xml_tag_integer(self):
        assert _extract_score("<score>7</score>") == 7.0

    def test_fallback_band_score_pattern(self):
        assert _extract_score("Band Score: 6.0") == 6.0

    def test_xml_preferred_over_body_pattern(self):
        assert _extract_score("<score>5.0</score>\nBand Score: 7.0") == 5.0

    def test_out_of_range_returns_none(self):
        assert _extract_score("<score>10.0</score>") is None

    def test_no_score_returns_none(self):
        assert _extract_score("No score anywhere in this text.") is None

    def test_zero_valid(self):
        assert _extract_score("<score>0</score>") == 0.0

    def test_nine_valid(self):
        assert _extract_score("<score>9</score>") == 9.0


# ---------------------------------------------------------------------------
# _extract_tag
# ---------------------------------------------------------------------------

class TestExtractTag:
    def test_simple(self):
        assert _extract_tag("<confidence>high</confidence>", "confidence") == "high"

    def test_strips_whitespace(self):
        assert _extract_tag("<confidence>\n  high\n</confidence>", "confidence") == "high"

    def test_missing_returns_none(self):
        assert _extract_tag("no tag here", "confidence") is None

    def test_different_tag_name(self):
        assert _extract_tag("<quality>strong</quality>", "quality") == "strong"

    def test_case_insensitive(self):
        assert _extract_tag("<Confidence>medium</Confidence>", "confidence") == "medium"

    def test_does_not_match_other_tags(self):
        assert _extract_tag("<score>6.5</score>", "confidence") is None


# ---------------------------------------------------------------------------
# generate_feedback
# ---------------------------------------------------------------------------

_STUB_RESPONSE = (
    "<score>6.5</score>\n"
    "<confidence>high</confidence>\n"
    "The essay addresses the task with clear arguments and sufficient detail."
)


class TestGenerateFeedback:
    @pytest.fixture
    def stub(self, stub_provider):
        stub_provider.set(_STUB_RESPONSE)
        return stub_provider

    def test_returns_all_four_components(self, stub):
        result = asyncio.run(generate_feedback(
            question="Some IELTS question",
            essay="Some essay text",
            policies={},
            strategy=StrategyGuidance(),
            provider=stub,
            model_config=ModelConfig(),
        ))
        assert set(result.keys()) == {"task_response", "coherence", "lexical", "grammar"}

    def test_each_result_is_component_feedback(self, stub):
        result = asyncio.run(generate_feedback(
            question="Q", essay="E", policies={},
            strategy=StrategyGuidance(), provider=stub, model_config=ModelConfig(),
        ))
        for component, fb in result.items():
            assert isinstance(fb, ComponentFeedback)
            assert fb.component == component

    def test_score_parsed_correctly(self, stub):
        result = asyncio.run(generate_feedback(
            question="Q", essay="E", policies={},
            strategy=StrategyGuidance(), provider=stub, model_config=ModelConfig(),
        ))
        for fb in result.values():
            assert fb.score == 6.5

    def test_tag_parsed_correctly(self, stub):
        result = asyncio.run(generate_feedback(
            question="Q", essay="E", policies={},
            strategy=StrategyGuidance(), provider=stub, model_config=ModelConfig(),
        ))
        for fb in result.values():
            assert fb.tag == "high"

    def test_feedback_text_non_empty(self, stub):
        result = asyncio.run(generate_feedback(
            question="Q", essay="E", policies={},
            strategy=StrategyGuidance(), provider=stub, model_config=ModelConfig(),
        ))
        for fb in result.values():
            assert len(fb.feedback.strip()) > 0

    def test_strategy_guidance_accepted(self, stub):
        strategy = StrategyGuidance(
            contrastive_anchoring=True,
            per_component_focus={"lexical": "Focus on vocabulary range and AoA"},
        )
        result = asyncio.run(generate_feedback(
            question="Q", essay="E", policies={},
            strategy=strategy, provider=stub, model_config=ModelConfig(),
        ))
        assert set(result.keys()) == {"task_response", "coherence", "lexical", "grammar"}

    def test_custom_policy_per_component(self, stub):
        policies = {"task_response": PromptPolicy(specificity=5, evidence_required=True)}
        result = asyncio.run(generate_feedback(
            question="Q", essay="E", policies=policies,
            strategy=StrategyGuidance(), provider=stub, model_config=ModelConfig(),
        ))
        assert "task_response" in result

    def test_missing_score_in_stub_gives_none(self, stub_provider):
        stub_provider.set("No structured output here, just prose.")
        result = asyncio.run(generate_feedback(
            question="Q", essay="E", policies={},
            strategy=StrategyGuidance(), provider=stub_provider, model_config=ModelConfig(),
        ))
        for fb in result.values():
            assert fb.score is None


# ---------------------------------------------------------------------------
# run_examiner_pass (tool function called directly, no HTTP)
# ---------------------------------------------------------------------------

_SMALL_TRAIN = pd.DataFrame([
    {"question": "Should cities ban cars?", "essay": f"Essay number {i}.", "band": "6.0"}
    for i in range(3)
])


@pytest.fixture
def session_env(tmp_path, stub_provider):
    """Minimal on-disk session + patched provider."""
    stub_provider.set(_STUB_RESPONSE)
    drive_path = tmp_path / "drive"

    state = SessionState(session_id="test-session")
    save_state(state, drive_path)

    data_dir = drive_path / "sessions" / "test-session" / "data"
    data_dir.mkdir(parents=True)
    _SMALL_TRAIN.to_parquet(data_dir / "train.parquet", index=False)

    return drive_path, stub_provider


def _make_tool_fn(stub_provider, monkeypatch, drive_path):
    import edsmith.examiner.mcp.tools as tools_mod
    from fastmcp import FastMCP

    monkeypatch.setenv("EDSMITH_DRIVE_PATH", str(drive_path))
    monkeypatch.setattr(tools_mod, "OpenRouterProvider", lambda: stub_provider)
    return tools_mod.register_examiner_pass(FastMCP("test"))


class TestRunExaminerPass:
    def test_parquet_written(self, session_env, monkeypatch):
        drive_path, stub = session_env
        fn = _make_tool_fn(stub, monkeypatch, drive_path)
        asyncio.run(fn(session_id="test-session", iteration=0))
        assert (drive_path / "sessions" / "test-session" / "feedback_iter0.parquet").exists()

    def test_summary_has_required_keys(self, session_env, monkeypatch):
        drive_path, stub = session_env
        fn = _make_tool_fn(stub, monkeypatch, drive_path)
        result = asyncio.run(fn(session_id="test-session", iteration=0))
        for key in ("session_id", "iteration", "essays_processed", "essays_total",
                    "components_covered", "score_distributions", "warnings", "parquet_path"):
            assert key in result

    def test_all_essays_processed(self, session_env, monkeypatch):
        drive_path, stub = session_env
        fn = _make_tool_fn(stub, monkeypatch, drive_path)
        result = asyncio.run(fn(session_id="test-session", iteration=0))
        assert result["essays_processed"] == 3
        assert result["essays_total"] == 3

    def test_all_components_covered(self, session_env, monkeypatch):
        drive_path, stub = session_env
        fn = _make_tool_fn(stub, monkeypatch, drive_path)
        result = asyncio.run(fn(session_id="test-session", iteration=0))
        assert result["components_covered"] == 3

    def test_score_distributions_present(self, session_env, monkeypatch):
        drive_path, stub = session_env
        fn = _make_tool_fn(stub, monkeypatch, drive_path)
        result = asyncio.run(fn(session_id="test-session", iteration=0))
        for component in ("task_response", "coherence", "lexical", "grammar"):
            assert component in result["score_distributions"]
            dist = result["score_distributions"][component]
            assert "mean" in dist and "std" in dist

    def test_parquet_has_correct_schema(self, session_env, monkeypatch):
        drive_path, stub = session_env
        fn = _make_tool_fn(stub, monkeypatch, drive_path)
        asyncio.run(fn(session_id="test-session", iteration=0))
        df = pd.read_parquet(drive_path / "sessions" / "test-session" / "feedback_iter0.parquet")
        for col in ("question", "essay", "component", "feedback_text", "score", "tag"):
            assert col in df.columns
        assert len(df) == 3 * 4  # 3 essays × 4 components
