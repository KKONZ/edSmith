from pathlib import Path

import pytest

from edsmith.config.session import (
    DiagnosticReport,
    HumanReviewProposal,
    PromptPolicy,
    StrategyGuidance,
)
from edsmith.session.state import (
    SessionMetrics,
    SessionState,
    load_metrics,
    load_proposal,
    load_state,
    save_metrics,
    save_proposal,
    save_state,
)


class TestSessionStateRoundtrip:
    def test_save_and_load(self, tmp_path):
        state = SessionState(session_id="abc123", iteration=2)
        save_state(state, tmp_path)
        loaded = load_state(tmp_path, "abc123")
        assert loaded.session_id == "abc123"
        assert loaded.iteration == 2

    def test_policies_preserved(self, tmp_path):
        state = SessionState(
            session_id="abc123",
            policies={"grammar": PromptPolicy(specificity=4)},
        )
        save_state(state, tmp_path)
        loaded = load_state(tmp_path, "abc123")
        assert loaded.policies["grammar"].specificity == 4

    def test_strategy_guidance_preserved(self, tmp_path):
        state = SessionState(
            session_id="abc123",
            strategy_guidance=StrategyGuidance(use_aoa=True, contrastive_anchoring=True),
        )
        save_state(state, tmp_path)
        loaded = load_state(tmp_path, "abc123")
        assert loaded.strategy_guidance.use_aoa is True
        assert loaded.strategy_guidance.contrastive_anchoring is True

    def test_parent_session_id_preserved(self, tmp_path):
        state = SessionState(session_id="child", parent_session_id="parent")
        save_state(state, tmp_path)
        loaded = load_state(tmp_path, "child")
        assert loaded.parent_session_id == "parent"

    def test_missing_state_raises(self, tmp_path):
        with pytest.raises(Exception):
            load_state(tmp_path, "nonexistent")


class TestSessionMetricsRoundtrip:
    def test_save_and_load(self, tmp_path):
        metrics = SessionMetrics(
            session_id="abc123",
            iteration=1,
            val={"accuracy": 0.48, "adjacent_accuracy": 0.81, "qwk": 0.71, "smd": 0.21},
            test={"accuracy": 0.44, "qwk": 0.68},
        )
        save_metrics(metrics, tmp_path)
        loaded = load_metrics(tmp_path, "abc123", 1)
        assert loaded.session_id == "abc123"
        assert loaded.iteration == 1
        assert loaded.val["accuracy"] == pytest.approx(0.48)
        assert loaded.test["qwk"] == pytest.approx(0.68)

    def test_file_location(self, tmp_path):
        metrics = SessionMetrics(
            session_id="s1", iteration=3,
            val={"accuracy": 0.5}, test={"accuracy": 0.45},
        )
        path = save_metrics(metrics, tmp_path)
        assert path == tmp_path / "sessions" / "s1" / "metrics_iter3.json"

    def test_multiple_iterations_independent(self, tmp_path):
        for i in range(3):
            m = SessionMetrics(
                session_id="s1", iteration=i,
                val={"accuracy": 0.4 + i * 0.05}, test={"accuracy": 0.38 + i * 0.05},
            )
            save_metrics(m, tmp_path)
        loaded = load_metrics(tmp_path, "s1", 2)
        assert loaded.val["accuracy"] == pytest.approx(0.50)

    def test_missing_metrics_raises(self, tmp_path):
        with pytest.raises(Exception):
            load_metrics(tmp_path, "nonexistent", 0)


class TestProposalRoundtrip:
    def test_save_and_load(self, tmp_path):
        report = DiagnosticReport(
            session_id="s1", iteration=0, summary="Test diagnostic"
        )
        proposal = HumanReviewProposal(
            session_id="s1", iteration=0, diagnostic_report=report
        )
        save_proposal(proposal, tmp_path)
        loaded = load_proposal(tmp_path, "s1", 0)
        assert loaded.session_id == "s1"
        assert loaded.status == "pending"
        assert loaded.diagnostic_report.summary == "Test diagnostic"

    def test_status_mutation_persisted(self, tmp_path):
        report = DiagnosticReport(session_id="s1", iteration=0)
        proposal = HumanReviewProposal(
            session_id="s1", iteration=0, diagnostic_report=report
        )
        save_proposal(proposal, tmp_path)
        loaded = load_proposal(tmp_path, "s1", 0)
        loaded.status = "approved"
        save_proposal(loaded, tmp_path)
        reloaded = load_proposal(tmp_path, "s1", 0)
        assert reloaded.status == "approved"

    def test_critique_preserved(self, tmp_path):
        report = DiagnosticReport(session_id="s1", iteration=0)
        proposal = HumanReviewProposal(
            session_id="s1", iteration=0,
            diagnostic_report=report,
            status="rejected",
            critique="Too many changes at once.",
        )
        save_proposal(proposal, tmp_path)
        loaded = load_proposal(tmp_path, "s1", 0)
        assert loaded.critique == "Too many changes at once."
        assert loaded.status == "rejected"
