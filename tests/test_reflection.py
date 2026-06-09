import math
from unittest.mock import MagicMock

import pytest

from edsmith.agents.phase2.reflection import (
    ReflectionAgent,
    ReflectionOutput,
    _extract_tag,
    _iteration_table,
    _parse_policy_updates,
    _policies_block,
    _ucb1,
)
from edsmith.config.session import PromptPolicy
from edsmith.memory.episodic import EpisodicRecord, IterationMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    session_id: str = "abc",
    parent_id: str | None = None,
    visit_count: int = 0,
    value_estimate: float = 0.0,
    iterations: list[IterationMetrics] | None = None,
    tree_depth: int = 0,
) -> EpisodicRecord:
    return EpisodicRecord(
        session_id=session_id,
        parent_session_id=parent_id,
        visit_count=visit_count,
        value_estimate=value_estimate,
        iterations=iterations or [],
        tree_depth=tree_depth,
    )


def _default_policies() -> dict[str, PromptPolicy]:
    return {c: PromptPolicy() for c in ("task_response", "coherence", "lexical", "grammar")}


# ---------------------------------------------------------------------------
# _ucb1
# ---------------------------------------------------------------------------

class TestUCB1:
    def test_unvisited_node_returns_inf(self):
        record = _make_record(visit_count=0)
        assert _ucb1(record, total_visits=10) == float("inf")

    def test_visited_node_matches_formula(self):
        record = _make_record(visit_count=3, value_estimate=0.6)
        total = 9
        c = math.sqrt(2)
        expected = 0.6 + c * math.sqrt(math.log(total) / 3)
        assert _ucb1(record, total_visits=total) == pytest.approx(expected, abs=1e-9)

    def test_higher_visit_count_lower_exploration_bonus(self):
        low_visits = _make_record(visit_count=1, value_estimate=0.5)
        high_visits = _make_record(visit_count=10, value_estimate=0.5)
        total = 20
        assert _ucb1(low_visits, total) > _ucb1(high_visits, total)


# ---------------------------------------------------------------------------
# _extract_tag (reflection version — same logic as feedback)
# ---------------------------------------------------------------------------

class TestExtractTag:
    def test_notes_tag(self):
        content = "<notes>Analysis here.</notes>"
        assert _extract_tag(content, "notes") == "Analysis here."

    def test_missing_tag_empty(self):
        assert _extract_tag("no tags", "notes") == ""

    def test_multiline_content(self):
        content = "<action>\nChange specificity.\n</action>"
        assert _extract_tag(content, "action") == "Change specificity."


# ---------------------------------------------------------------------------
# _parse_policy_updates
# ---------------------------------------------------------------------------

class TestParsePolicyUpdates:
    def test_empty_policies_tag_returns_current(self):
        current = _default_policies()
        result = _parse_policy_updates("<policies></policies>", current)
        for comp in current:
            assert result[comp] == current[comp]

    def test_missing_policies_tag_returns_current(self):
        current = _default_policies()
        result = _parse_policy_updates("no policies here", current)
        assert result == current

    def test_invalid_json_returns_current(self):
        current = _default_policies()
        result = _parse_policy_updates("<policies>{bad json}</policies>", current)
        assert result == current

    def test_partial_update_merged(self):
        current = _default_policies()
        content = '<policies>{"task_response": {"specificity": 4}}</policies>'
        result = _parse_policy_updates(content, current)
        assert result["task_response"].specificity == 4
        # other components unchanged
        assert result["coherence"].specificity == current["coherence"].specificity

    def test_unknown_field_ignored(self):
        current = _default_policies()
        content = '<policies>{"grammar": {"nonexistent_field": 99}}</policies>'
        result = _parse_policy_updates(content, current)
        assert result["grammar"] == current["grammar"]

    def test_bad_value_keeps_existing(self):
        current = _default_policies()
        # specificity must be 1-5; 99 is invalid → Pydantic raises → keep existing
        content = '<policies>{"lexical": {"specificity": 99}}</policies>'
        result = _parse_policy_updates(content, current)
        assert result["lexical"] == current["lexical"]

    def test_unknown_component_ignored(self):
        current = _default_policies()
        content = '<policies>{"nonexistent": {"specificity": 3}}</policies>'
        result = _parse_policy_updates(content, current)
        assert set(result.keys()) == set(current.keys())

    def test_multiple_components_updated(self):
        current = _default_policies()
        content = '<policies>{"task_response": {"specificity": 5}, "grammar": {"evidence_required": false}}</policies>'
        result = _parse_policy_updates(content, current)
        assert result["task_response"].specificity == 5
        assert result["grammar"].evidence_required is False


# ---------------------------------------------------------------------------
# _iteration_table
# ---------------------------------------------------------------------------

class TestIterationTable:
    def test_empty_iterations(self):
        record = _make_record()
        assert _iteration_table(record) == "No iteration data."

    def test_non_empty_contains_metrics(self):
        record = _make_record(iterations=[
            IterationMetrics(iteration=1, accuracy=0.45, adjacent_accuracy=0.80, qwk=0.72, smd=0.1),
        ])
        table = _iteration_table(record)
        assert "0.4500" in table
        assert "0.8000" in table

    def test_header_row_present(self):
        record = _make_record(iterations=[
            IterationMetrics(iteration=1, accuracy=0.5, adjacent_accuracy=0.8, qwk=0.7, smd=0.0),
        ])
        table = _iteration_table(record)
        assert "accuracy" in table
        assert "qwk" in table


# ---------------------------------------------------------------------------
# _policies_block
# ---------------------------------------------------------------------------

class TestPoliciesBlock:
    def test_contains_all_components(self):
        policies = _default_policies()
        block = _policies_block(policies)
        for comp in policies:
            assert comp in block

    def test_contains_header_row(self):
        block = _policies_block(_default_policies())
        assert "specificity" in block
        assert "evidence_required" in block


# ---------------------------------------------------------------------------
# ReflectionAgent._select_mode
# ---------------------------------------------------------------------------

class TestSelectMode:
    def _make_agent(self, *, mcts_eligible: bool, beam_eligible: bool, stub_provider):
        mock_mem = MagicMock()
        mock_mem.mcts_eligible.return_value = mcts_eligible
        mock_mem.beam_search_eligible.return_value = beam_eligible
        return ReflectionAgent(
            provider=stub_provider,
            episodic_memory=mock_mem,
            model="stub",
        )

    def test_simple_mode(self, stub_provider):
        agent = self._make_agent(mcts_eligible=False, beam_eligible=False, stub_provider=stub_provider)
        record = _make_record()
        assert agent._select_mode(record) == "simple"

    def test_beam_mode(self, stub_provider):
        agent = self._make_agent(mcts_eligible=False, beam_eligible=True, stub_provider=stub_provider)
        record = _make_record()
        assert agent._select_mode(record) == "beam"

    def test_mcts_mode_takes_priority_over_beam(self, stub_provider):
        agent = self._make_agent(mcts_eligible=True, beam_eligible=True, stub_provider=stub_provider)
        record = _make_record()
        assert agent._select_mode(record) == "mcts"


# ---------------------------------------------------------------------------
# ReflectionAgent.reflect (end-to-end with stub provider)
# ---------------------------------------------------------------------------

_STUB_RESPONSE = """\
<notes>Accuracy was low on task_response. Specificity needs to increase.</notes>
<action>Raise task_response specificity to 4.</action>
<policies>{"task_response": {"specificity": 4}}</policies>
"""


class TestReflectionAgentReflect:
    def _make_simple_agent(self, stub_provider):
        mock_mem = MagicMock()
        mock_mem.mcts_eligible.return_value = False
        mock_mem.beam_search_eligible.return_value = False
        stub_provider.set(_STUB_RESPONSE)
        return ReflectionAgent(provider=stub_provider, episodic_memory=mock_mem, model="stub")

    def test_reflect_returns_output(self, stub_provider):
        agent = self._make_simple_agent(stub_provider)
        record = _make_record(iterations=[
            IterationMetrics(iteration=1, accuracy=0.3, adjacent_accuracy=0.6, qwk=0.5, smd=0.05),
        ])
        out = agent.reflect(record, _default_policies(), val_summary={"accuracy": 0.3})
        assert isinstance(out, ReflectionOutput)

    def test_reflect_mode_is_simple(self, stub_provider):
        agent = self._make_simple_agent(stub_provider)
        record = _make_record()
        out = agent.reflect(record, _default_policies(), val_summary={"accuracy": 0.3})
        assert out.mode == "simple"

    def test_reflect_notes_parsed(self, stub_provider):
        agent = self._make_simple_agent(stub_provider)
        record = _make_record()
        out = agent.reflect(record, _default_policies(), val_summary={"accuracy": 0.3})
        assert "task_response" in out.notes

    def test_reflect_action_parsed(self, stub_provider):
        agent = self._make_simple_agent(stub_provider)
        record = _make_record()
        out = agent.reflect(record, _default_policies(), val_summary={"accuracy": 0.3})
        assert "specificity" in out.action_taken.lower()

    def test_reflect_policy_updated(self, stub_provider):
        agent = self._make_simple_agent(stub_provider)
        record = _make_record()
        out = agent.reflect(record, _default_policies(), val_summary={"accuracy": 0.3})
        assert out.suggested_policies["task_response"].specificity == 4
