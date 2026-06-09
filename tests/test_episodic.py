import pytest

from edsmith.memory.episodic import (
    EpisodicMemory,
    EpisodicRecord,
    IterationMetrics,
    _deserialize,
    _serialize,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    session_id: str = "abc",
    parent_id: str | None = None,
    tree_depth: int = 0,
    visit_count: int = 1,
    value_estimate: float = 0.5,
    iterations: list[IterationMetrics] | None = None,
    reflection_notes: str = "",
) -> EpisodicRecord:
    return EpisodicRecord(
        session_id=session_id,
        parent_session_id=parent_id,
        tree_depth=tree_depth,
        visit_count=visit_count,
        value_estimate=value_estimate,
        iterations=iterations or [],
        reflection_notes=reflection_notes,
    )


# ---------------------------------------------------------------------------
# Serialisation roundtrip
# ---------------------------------------------------------------------------

class TestSerialiseDeserialise:
    def test_roundtrip_minimal(self):
        record = _make_record()
        text = _serialize(record)
        recovered = _deserialize(text)
        assert recovered.session_id == record.session_id
        assert recovered.visit_count == record.visit_count
        assert recovered.value_estimate == record.value_estimate

    def test_roundtrip_with_iterations(self):
        iters = [
            IterationMetrics(iteration=1, accuracy=0.4, adjacent_accuracy=0.75, qwk=0.6, smd=0.05),
            IterationMetrics(iteration=2, accuracy=0.5, adjacent_accuracy=0.80, qwk=0.65, smd=0.02),
        ]
        record = _make_record(iterations=iters)
        recovered = _deserialize(_serialize(record))
        assert len(recovered.iterations) == 2
        assert recovered.iterations[1].accuracy == pytest.approx(0.5)

    def test_roundtrip_reflection_notes(self):
        record = _make_record(reflection_notes="Accuracy improved significantly.")
        recovered = _deserialize(_serialize(record))
        assert "improved" in recovered.reflection_notes

    def test_invalid_text_raises(self):
        with pytest.raises(ValueError, match="Invalid episodic record"):
            _deserialize("no frontmatter here")

    def test_parent_session_id_preserved(self):
        record = _make_record(session_id="child", parent_id="parent", tree_depth=1)
        recovered = _deserialize(_serialize(record))
        assert recovered.parent_session_id == "parent"
        assert recovered.tree_depth == 1


# ---------------------------------------------------------------------------
# EpisodicMemory — save / load
# ---------------------------------------------------------------------------

class TestEpisodicMemorySaveLoad:
    def test_save_and_load(self, tmp_path):
        mem = EpisodicMemory(drive_path=tmp_path)
        record = _make_record(session_id="sess1")
        mem.save(record)
        loaded = mem.load("sess1")
        assert loaded.session_id == "sess1"

    def test_load_missing_raises(self, tmp_path):
        mem = EpisodicMemory(drive_path=tmp_path)
        with pytest.raises(FileNotFoundError):
            mem.load("nonexistent")

    def test_save_overwrites(self, tmp_path):
        mem = EpisodicMemory(drive_path=tmp_path)
        record = _make_record(session_id="sess1", reflection_notes="v1")
        mem.save(record)
        record2 = _make_record(session_id="sess1", reflection_notes="v2")
        mem.save(record2)
        loaded = mem.load("sess1")
        assert "v2" in loaded.reflection_notes

    def test_load_all_empty(self, tmp_path):
        mem = EpisodicMemory(drive_path=tmp_path)
        assert mem.load_all() == []

    def test_load_all_returns_all_records(self, tmp_path):
        mem = EpisodicMemory(drive_path=tmp_path)
        for sid in ("a", "b", "c"):
            mem.save(_make_record(session_id=sid))
        records = mem.load_all()
        assert {r.session_id for r in records} == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# Tree queries
# ---------------------------------------------------------------------------

class TestTreeQueries:
    def _setup_tree(self, tmp_path) -> EpisodicMemory:
        mem = EpisodicMemory(drive_path=tmp_path)
        # root
        mem.save(_make_record(session_id="root", parent_id=None, tree_depth=0))
        # two children of root
        mem.save(_make_record(session_id="child1", parent_id="root", tree_depth=1))
        mem.save(_make_record(session_id="child2", parent_id="root", tree_depth=1))
        # one grandchild
        mem.save(_make_record(session_id="gc1", parent_id="child1", tree_depth=2))
        return mem

    def test_get_children(self, tmp_path):
        mem = self._setup_tree(tmp_path)
        children = mem.get_children("root")
        assert {r.session_id for r in children} == {"child1", "child2"}

    def test_get_children_leaf_empty(self, tmp_path):
        mem = self._setup_tree(tmp_path)
        assert mem.get_children("gc1") == []

    def test_get_siblings(self, tmp_path):
        mem = self._setup_tree(tmp_path)
        siblings = mem.get_siblings("child1")
        assert [r.session_id for r in siblings] == ["child2"]

    def test_get_siblings_root_has_none(self, tmp_path):
        mem = self._setup_tree(tmp_path)
        assert mem.get_siblings("root") == []

    def test_get_nodes_at_depth(self, tmp_path):
        mem = self._setup_tree(tmp_path)
        depth1 = mem.get_nodes_at_depth(1)
        assert {r.session_id for r in depth1} == {"child1", "child2"}


# ---------------------------------------------------------------------------
# Eligibility checks
# ---------------------------------------------------------------------------

class TestEligibility:
    def test_beam_eligible_with_two_siblings(self, tmp_path):
        mem = EpisodicMemory(drive_path=tmp_path)
        mem.save(_make_record(session_id="root", parent_id=None))
        mem.save(_make_record(session_id="s1", parent_id="root"))
        mem.save(_make_record(session_id="s2", parent_id="root"))
        mem.save(_make_record(session_id="s3", parent_id="root"))
        # s1 has siblings s2 and s3 → ≥2 siblings → eligible
        assert mem.beam_search_eligible("s1") is True

    def test_beam_not_eligible_with_one_sibling(self, tmp_path):
        mem = EpisodicMemory(drive_path=tmp_path)
        mem.save(_make_record(session_id="root", parent_id=None))
        mem.save(_make_record(session_id="s1", parent_id="root"))
        mem.save(_make_record(session_id="s2", parent_id="root"))
        # s1 has only s2 → 1 sibling → not eligible
        assert mem.beam_search_eligible("s1") is False

    def test_mcts_eligible_with_five_nodes(self, tmp_path):
        mem = EpisodicMemory(drive_path=tmp_path)
        for i in range(5):
            mem.save(_make_record(session_id=str(i)))
        assert mem.mcts_eligible(min_nodes=5) is True

    def test_mcts_not_eligible_with_four_nodes(self, tmp_path):
        mem = EpisodicMemory(drive_path=tmp_path)
        for i in range(4):
            mem.save(_make_record(session_id=str(i)))
        assert mem.mcts_eligible(min_nodes=5) is False


# ---------------------------------------------------------------------------
# EpisodicRecord properties
# ---------------------------------------------------------------------------

class TestEpisodicRecordProperties:
    def test_best_accuracy_no_iterations(self):
        record = _make_record()
        assert record.best_accuracy is None

    def test_best_accuracy_with_iterations(self):
        record = _make_record(iterations=[
            IterationMetrics(iteration=1, accuracy=0.4, adjacent_accuracy=0.7, qwk=0.6, smd=0.0),
            IterationMetrics(iteration=2, accuracy=0.6, adjacent_accuracy=0.8, qwk=0.7, smd=0.0),
        ])
        assert record.best_accuracy == pytest.approx(0.6)

    def test_final_metrics_no_iterations(self):
        assert _make_record().final_metrics is None

    def test_final_metrics_last_iteration(self):
        record = _make_record(iterations=[
            IterationMetrics(iteration=1, accuracy=0.4, adjacent_accuracy=0.7, qwk=0.6, smd=0.0),
            IterationMetrics(iteration=2, accuracy=0.6, adjacent_accuracy=0.8, qwk=0.7, smd=0.0),
        ])
        assert record.final_metrics.iteration == 2
