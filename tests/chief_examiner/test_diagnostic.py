import asyncio
import json
from pathlib import Path

import pandas as pd
import pytest

from edsmith.chief_examiner.diagnostic import (
    _feedback_sample,
    _load_iteration_history,
    _parse_response,
    run_diagnostic,
)
from edsmith.config.session import (
    DiagnosticReport,
    HumanReviewProposal,
    ModelConfig,
    PromptPolicy,
    StrategyGuidance,
)
from edsmith.session.state import (
    SessionMetrics,
    SessionState,
    save_metrics,
    save_proposal,
    save_state,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_COMPONENTS = ["task_response", "coherence", "lexical", "grammar"]

_SMALL_FEEDBACK = pd.DataFrame([
    {
        "question": "Should cities ban cars?",
        "essay": f"Essay {i}. This is a sample essay about urban transport.",
        "band": "6.0",
        "component": c,
        "feedback_text": f"Feedback for {c} on essay {i}.",
        "score": 6.0,
        "tag": "high",
    }
    for i in range(4)
    for c in _COMPONENTS
])

_STUB_DIAGNOSTIC = json.dumps({
    "summary": "Lexical feedback is too generic; grammar errors on basic words under-penalised.",
    "per_component_issues": {
        "task_response": "Adequate",
        "coherence": "Adequate",
        "lexical": "Feedback describes vocabulary as varied without checking AoA distribution.",
        "grammar": "Basic errors (AoA < 5) are under-penalised.",
    },
    "proposed_strategy": {
        "use_aoa": True,
        "use_grammar": True,
        "contrastive_anchoring": False,
        "per_component_focus": {"lexical": "Focus on AoA mean and pct_late"},
    },
    "proposed_policies": {
        "lexical": {"specificity": 4, "evidence_required": True},
        "grammar": {"evidence_required": True, "additional_instructions": "Penalise basic errors harshly."},
    },
})

_STUB_RESPONSE = f"<diagnostic>{_STUB_DIAGNOSTIC}</diagnostic>"


# ---------------------------------------------------------------------------
# _feedback_sample
# ---------------------------------------------------------------------------

_DIVERGENT_FEEDBACK = pd.DataFrame([
    {
        "question": "Q", "essay": f"Essay {i}",
        "band": str(band), "component": c,
        "feedback_text": label,
        "score": score, "tag": tag,
    }
    for i, (c, band, score, label, tag) in enumerate([
        ("task_response", 5.0, 8.0, "Outlier feedback task_response.", "low"),
        ("task_response", 6.0, 6.0, "Normal feedback task_response.", "high"),
        ("coherence",     5.0, 8.0, "Outlier feedback coherence.", "low"),
        ("coherence",     6.0, 6.0, "Normal feedback coherence.", "high"),
        ("lexical",       5.0, 8.0, "Outlier feedback lexical.", "low"),
        ("lexical",       6.0, 6.0, "Normal feedback lexical.", "high"),
        ("grammar",       5.0, 8.0, "Outlier feedback grammar.", "low"),
        ("grammar",       6.0, 6.0, "Normal feedback grammar.", "high"),
    ])
])


class TestFeedbackSample:
    def test_includes_predicted_score(self):
        result = _feedback_sample(_SMALL_FEEDBACK)
        assert "predicted=" in result

    def test_includes_reference_band(self):
        result = _feedback_sample(_SMALL_FEEDBACK)
        assert "reference=" in result

    def test_includes_delta(self):
        result = _feedback_sample(_SMALL_FEEDBACK)
        assert "Δ=" in result

    def test_includes_confidence_tag(self):
        result = _feedback_sample(_SMALL_FEEDBACK)
        assert "confidence=high" in result

    def test_covers_all_four_components(self):
        result = _feedback_sample(_SMALL_FEEDBACK)
        for c in ["task_response", "coherence", "lexical", "grammar"]:
            assert c in result

    def test_outlier_sorted_before_normal(self):
        result = _feedback_sample(_DIVERGENT_FEEDBACK)
        assert result.index("Outlier") < result.index("Normal")

    def test_handles_missing_band_gracefully(self):
        df = _SMALL_FEEDBACK.copy()
        df["band"] = None
        result = _feedback_sample(df)
        assert "predicted=" in result

    def test_handles_empty_dataframe(self):
        result = _feedback_sample(pd.DataFrame(columns=_SMALL_FEEDBACK.columns))
        assert result == ""


# ---------------------------------------------------------------------------
# _load_iteration_history
# ---------------------------------------------------------------------------

class TestLoadIterationHistory:
    def test_first_iteration_returns_no_prior_message(self, tmp_path):
        result = _load_iteration_history(tmp_path, "s1", 0)
        assert "first diagnostic" in result.lower()

    def test_history_empty_when_files_missing(self, tmp_path):
        # iteration=1 but no files on disk — should not raise
        result = _load_iteration_history(tmp_path, "s1", 1)
        assert "Iteration 0" in result

    def test_metrics_included_when_present(self, tmp_path):
        m = SessionMetrics(
            session_id="s1", iteration=0,
            val={"accuracy": 0.48, "qwk": 0.71, "smd": 0.21, "adjacent_accuracy": 0.81},
            test={"accuracy": 0.44},
        )
        save_metrics(m, tmp_path)
        result = _load_iteration_history(tmp_path, "s1", 1)
        assert "0.480" in result
        assert "0.710" in result

    def test_approved_proposal_in_history(self, tmp_path):
        _write_approved_proposal(tmp_path, "s1", 0)
        result = _load_iteration_history(tmp_path, "s1", 1)
        assert "approved" in result

    def test_rejected_proposal_shows_critique(self, tmp_path):
        report = DiagnosticReport(session_id="s1", iteration=0, summary="Test")
        proposal = HumanReviewProposal(
            session_id="s1", iteration=0,
            diagnostic_report=report,
            status="rejected",
            critique="Too many changes at once.",
        )
        save_proposal(proposal, tmp_path)
        result = _load_iteration_history(tmp_path, "s1", 1)
        assert "rejected" in result
        assert "Too many changes" in result

    def test_multiple_iterations_all_present(self, tmp_path):
        for i in range(3):
            m = SessionMetrics(
                session_id="s1", iteration=i,
                val={"accuracy": 0.4 + i * 0.05, "qwk": 0.6 + i * 0.03,
                     "smd": 0.1, "adjacent_accuracy": 0.75},
                test={"accuracy": 0.38},
            )
            save_metrics(m, tmp_path)
        result = _load_iteration_history(tmp_path, "s1", 3)
        assert "Iteration 0" in result
        assert "Iteration 1" in result
        assert "Iteration 2" in result


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:
    @pytest.fixture
    def base_policies(self):
        return {c: PromptPolicy() for c in _COMPONENTS}

    def test_summary_extracted(self, base_policies):
        _, proposal = _parse_response(
            _STUB_RESPONSE, "s1", 0, base_policies,
            StrategyGuidance(), {}, {},
        )
        assert "generic" in proposal.diagnostic_report.summary.lower()

    def test_per_component_issues_extracted(self, base_policies):
        _, proposal = _parse_response(
            _STUB_RESPONSE, "s1", 0, base_policies,
            StrategyGuidance(), {}, {},
        )
        assert "lexical" in proposal.diagnostic_report.per_component_issues
        assert "AoA" in proposal.diagnostic_report.per_component_issues["lexical"]

    def test_proposed_strategy_merged_onto_current(self, base_policies):
        current = StrategyGuidance(use_complexity=True)
        _, proposal = _parse_response(
            _STUB_RESPONSE, "s1", 0, base_policies, current, {}, {},
        )
        assert proposal.proposed_strategy.use_aoa is True
        assert proposal.proposed_strategy.use_grammar is True
        # use_complexity was True in current and not changed by proposal → preserved
        assert proposal.proposed_strategy.use_complexity is True

    def test_proposed_policies_merged_onto_current(self, base_policies):
        _, proposal = _parse_response(
            _STUB_RESPONSE, "s1", 0, base_policies,
            StrategyGuidance(), {}, {},
        )
        assert proposal.proposed_policies["lexical"].specificity == 4
        assert proposal.proposed_policies["lexical"].evidence_required is True

    def test_unchanged_components_keep_current_policy(self, base_policies):
        # task_response and coherence are not in proposed_policies
        _, proposal = _parse_response(
            _STUB_RESPONSE, "s1", 0, base_policies,
            StrategyGuidance(), {}, {},
        )
        assert proposal.proposed_policies["task_response"].specificity == PromptPolicy().specificity

    def test_unknown_fields_in_json_ignored(self, base_policies):
        raw = json.loads(_STUB_DIAGNOSTIC)
        raw["unknown_field"] = "should be ignored"
        raw["proposed_strategy"]["nonexistent_flag"] = True
        text = f"<diagnostic>{json.dumps(raw)}</diagnostic>"
        _, proposal = _parse_response(
            text, "s1", 0, base_policies, StrategyGuidance(), {}, {},
        )
        assert not hasattr(proposal.proposed_strategy, "nonexistent_flag")

    def test_malformed_json_falls_back_to_defaults(self, base_policies):
        report, proposal = _parse_response(
            "<diagnostic>not valid json{</diagnostic>",
            "s1", 0, base_policies, StrategyGuidance(), {}, {},
        )
        assert report.summary == ""
        assert proposal.status == "pending"

    def test_missing_diagnostic_tag_falls_back_to_defaults(self, base_policies):
        report, proposal = _parse_response(
            "The Chief Examiner found some issues but forgot to format correctly.",
            "s1", 0, base_policies, StrategyGuidance(), {}, {},
        )
        assert report.summary == ""
        assert isinstance(proposal, HumanReviewProposal)

    def test_per_component_focus_preserved(self, base_policies):
        _, proposal = _parse_response(
            _STUB_RESPONSE, "s1", 0, base_policies,
            StrategyGuidance(), {}, {},
        )
        assert "lexical" in proposal.proposed_strategy.per_component_focus

    def test_proposal_status_is_pending(self, base_policies):
        _, proposal = _parse_response(
            _STUB_RESPONSE, "s1", 0, base_policies,
            StrategyGuidance(), {}, {},
        )
        assert proposal.status == "pending"


# ---------------------------------------------------------------------------
# run_diagnostic (full async, StubProvider)
# ---------------------------------------------------------------------------

def _write_approved_proposal(drive_path: Path, session_id: str, iteration: int) -> None:
    report = DiagnosticReport(session_id=session_id, iteration=iteration, summary="Prior diagnosis")
    proposal = HumanReviewProposal(
        session_id=session_id, iteration=iteration,
        diagnostic_report=report,
        status="approved",
    )
    save_proposal(proposal, drive_path)


class TestRunDiagnostic:
    @pytest.fixture
    def session_env(self, tmp_path):
        state = SessionState(session_id="s1")
        save_state(state, tmp_path)
        return tmp_path

    def test_returns_report_and_proposal(self, session_env, stub_provider):
        stub_provider.set(_STUB_RESPONSE)
        report, proposal = asyncio.run(run_diagnostic(
            session_id="s1", iteration=0,
            feedback_df=_SMALL_FEEDBACK,
            val_metrics={"accuracy": 0.48, "qwk": 0.71, "smd": 0.21, "adjacent_accuracy": 0.81},
            test_metrics_summary={"accuracy": 0.44, "qwk": 0.68},
            policies={c: PromptPolicy() for c in _COMPONENTS},
            strategy=StrategyGuidance(),
            provider=stub_provider,
            model_config=ModelConfig(),
            drive_path=session_env,
        ))
        assert isinstance(report, DiagnosticReport)
        assert isinstance(proposal, HumanReviewProposal)

    def test_diagnostic_summary_populated(self, session_env, stub_provider):
        stub_provider.set(_STUB_RESPONSE)
        report, _ = asyncio.run(run_diagnostic(
            session_id="s1", iteration=0,
            feedback_df=_SMALL_FEEDBACK,
            val_metrics={"accuracy": 0.48, "qwk": 0.71, "smd": 0.21, "adjacent_accuracy": 0.81},
            test_metrics_summary={"accuracy": 0.44, "qwk": 0.68},
            policies={c: PromptPolicy() for c in _COMPONENTS},
            strategy=StrategyGuidance(),
            provider=stub_provider,
            model_config=ModelConfig(),
            drive_path=session_env,
        ))
        assert len(report.summary) > 0

    def test_first_iteration_no_history_files_needed(self, session_env, stub_provider):
        stub_provider.set(_STUB_RESPONSE)
        # No metrics or proposals on disk — should not raise
        report, proposal = asyncio.run(run_diagnostic(
            session_id="s1", iteration=0,
            feedback_df=_SMALL_FEEDBACK,
            val_metrics={"accuracy": 0.48, "qwk": 0.71, "smd": 0.21, "adjacent_accuracy": 0.81},
            test_metrics_summary={"accuracy": 0.44},
            policies={c: PromptPolicy() for c in _COMPONENTS},
            strategy=StrategyGuidance(),
            provider=stub_provider,
            model_config=ModelConfig(),
            drive_path=session_env,
        ))
        assert proposal.status == "pending"

    def test_history_loaded_for_second_iteration(self, session_env, stub_provider):
        stub_provider.set(_STUB_RESPONSE)
        m = SessionMetrics(
            session_id="s1", iteration=0,
            val={"accuracy": 0.42, "qwk": 0.61, "smd": 0.38, "adjacent_accuracy": 0.74},
            test={"accuracy": 0.39},
        )
        save_metrics(m, session_env)
        _write_approved_proposal(session_env, "s1", 0)

        # Run diagnostic at iteration 1 — should load history without error
        report, proposal = asyncio.run(run_diagnostic(
            session_id="s1", iteration=1,
            feedback_df=_SMALL_FEEDBACK,
            val_metrics={"accuracy": 0.48, "qwk": 0.71, "smd": 0.21, "adjacent_accuracy": 0.81},
            test_metrics_summary={"accuracy": 0.44},
            policies={c: PromptPolicy() for c in _COMPONENTS},
            strategy=StrategyGuidance(),
            provider=stub_provider,
            model_config=ModelConfig(),
            drive_path=session_env,
        ))
        assert isinstance(proposal, HumanReviewProposal)

    def test_critique_accepted_as_parameter(self, session_env, stub_provider):
        stub_provider.set(_STUB_RESPONSE)
        _, proposal = asyncio.run(run_diagnostic(
            session_id="s1", iteration=0,
            feedback_df=_SMALL_FEEDBACK,
            val_metrics={"accuracy": 0.48, "qwk": 0.71, "smd": 0.21, "adjacent_accuracy": 0.81},
            test_metrics_summary={"accuracy": 0.44},
            policies={c: PromptPolicy() for c in _COMPONENTS},
            strategy=StrategyGuidance(),
            provider=stub_provider,
            model_config=ModelConfig(),
            drive_path=session_env,
            critique="Focus only on the lexical component this iteration.",
        ))
        assert isinstance(proposal, HumanReviewProposal)

    def test_metric_summary_stored_in_report(self, session_env, stub_provider):
        stub_provider.set(_STUB_RESPONSE)
        val = {"accuracy": 0.48, "qwk": 0.71, "smd": 0.21, "adjacent_accuracy": 0.81}
        test = {"accuracy": 0.44, "qwk": 0.68}
        report, _ = asyncio.run(run_diagnostic(
            session_id="s1", iteration=0,
            feedback_df=_SMALL_FEEDBACK,
            val_metrics=val,
            test_metrics_summary=test,
            policies={c: PromptPolicy() for c in _COMPONENTS},
            strategy=StrategyGuidance(),
            provider=stub_provider,
            model_config=ModelConfig(),
            drive_path=session_env,
        ))
        assert "accuracy" in report.metric_summary
        assert "test_accuracy" in report.metric_summary
