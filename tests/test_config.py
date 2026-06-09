import pytest

from edsmith.config.session import (
    PromptPolicy,
    SamplingConfig,
    SessionConfig,
)
from edsmith.data.parser import COMPONENT_HEADINGS


# ---------------------------------------------------------------------------
# PromptPolicy
# ---------------------------------------------------------------------------

class TestPromptPolicy:
    def test_default_values(self):
        p = PromptPolicy()
        assert p.specificity == 2
        assert p.evidence_required is True
        assert p.feedback_granularity == "component"
        assert p.additional_instructions == ""

    def test_specificity_min_boundary(self):
        p = PromptPolicy(specificity=1)
        assert p.specificity == 1

    def test_specificity_max_boundary(self):
        p = PromptPolicy(specificity=5)
        assert p.specificity == 5

    def test_specificity_below_min_raises(self):
        with pytest.raises(Exception):
            PromptPolicy(specificity=0)

    def test_specificity_above_max_raises(self):
        with pytest.raises(Exception):
            PromptPolicy(specificity=6)

    def test_invalid_granularity_raises(self):
        with pytest.raises(Exception):
            PromptPolicy(feedback_granularity="invalid")


# ---------------------------------------------------------------------------
# SamplingConfig
# ---------------------------------------------------------------------------

class TestSamplingConfig:
    def test_default_ratios_sum_to_one(self):
        s = SamplingConfig()
        total = s.reflection_correct_ratio + s.reflection_adjacent_ratio + s.reflection_large_miss_ratio
        assert abs(total - 1.0) < 1e-9

    def test_custom_valid_ratios(self):
        s = SamplingConfig(
            reflection_correct_ratio=0.5,
            reflection_adjacent_ratio=0.3,
            reflection_large_miss_ratio=0.2,
        )
        assert s.reflection_correct_ratio == pytest.approx(0.5)

    def test_invalid_ratios_raise(self):
        with pytest.raises(Exception, match="sum to 1.0"):
            SamplingConfig(
                reflection_correct_ratio=0.5,
                reflection_adjacent_ratio=0.5,
                reflection_large_miss_ratio=0.5,
            )

    def test_validation_ratio_bounds(self):
        with pytest.raises(Exception):
            SamplingConfig(validation_ratio=0.0)
        with pytest.raises(Exception):
            SamplingConfig(validation_ratio=1.0)


# ---------------------------------------------------------------------------
# SessionConfig
# ---------------------------------------------------------------------------

class TestSessionConfig:
    def test_default_prompt_policies_cover_all_components(self):
        cfg = SessionConfig()
        assert set(cfg.prompt_policies.keys()) == set(COMPONENT_HEADINGS.keys())

    def test_custom_policy_merged_with_defaults(self):
        cfg = SessionConfig(
            prompt_policies={"task_response": {"specificity": 4}}
        )
        assert cfg.prompt_policies["task_response"].specificity == 4
        # other components still get defaults
        assert "coherence" in cfg.prompt_policies

    def test_n_iterations_min_one(self):
        with pytest.raises(Exception):
            SessionConfig(n_iterations=0)

    def test_yaml_roundtrip(self, tmp_path):
        cfg = SessionConfig(n_iterations=3)
        yaml_path = tmp_path / "session.yaml"
        cfg.to_yaml(yaml_path)
        loaded = SessionConfig.from_yaml(yaml_path)
        assert loaded.n_iterations == 3
        assert set(loaded.prompt_policies.keys()) == set(COMPONENT_HEADINGS.keys())

    def test_yaml_roundtrip_nested_config(self, tmp_path):
        cfg = SessionConfig()
        cfg.scorer.lora_r = 32
        yaml_path = tmp_path / "session.yaml"
        cfg.to_yaml(yaml_path)
        loaded = SessionConfig.from_yaml(yaml_path)
        assert loaded.scorer.lora_r == 32

    def test_from_yaml_missing_file_raises(self, tmp_path):
        with pytest.raises(Exception):
            SessionConfig.from_yaml(tmp_path / "nonexistent.yaml")
