from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    generator: str = "meta-llama/llama-3.1-8b-instruct"
    chair: str = "anthropic/claude-sonnet-4-5"
    enable_thinking: bool = False


class PromptPolicy(BaseModel):
    specificity: Annotated[int, Field(ge=1, le=5)] = 2
    evidence_required: bool = True
    feedback_granularity: Literal["component", "overall", "both"] = "component"
    additional_instructions: str = ""


class StrategyGuidance(BaseModel):
    """Higher-level strategic direction from Chief Examiner to Examiner."""
    per_component_focus: dict[str, str] = Field(default_factory=dict)
    use_aoa: bool = False
    use_grammar: bool = False
    use_complexity: bool = False
    use_discourse: bool = False
    contrastive_anchoring: bool = False
    use_tool_calling: bool = False  # when True, tools are offered as API calls; when False, pre-injected as text


class DiagnosticReport(BaseModel):
    """Chief Examiner's analysis of feedback quality against training data."""
    session_id: str
    iteration: int
    summary: str = ""
    per_component_issues: dict[str, str] = Field(default_factory=dict)
    metric_summary: dict[str, float] = Field(default_factory=dict)
    linguistic_findings: dict[str, Any] = Field(default_factory=dict)


class HumanReviewProposal(BaseModel):
    """Proposal surfaced to the human after Chief Examiner diagnostic."""
    session_id: str
    iteration: int
    diagnostic_report: DiagnosticReport
    proposed_strategy: StrategyGuidance = Field(default_factory=StrategyGuidance)
    proposed_policies: dict[str, PromptPolicy] = Field(default_factory=dict)
    status: Literal["pending", "approved", "rejected"] = "pending"
    critique: str | None = None


class SamplingConfig(BaseModel):
    size: int | None = None
    random_state: int = 42
    validation_ratio: Annotated[float, Field(gt=0, lt=1)] = 0.15
    test_ratio: Annotated[float, Field(gt=0, lt=1)] | None = None


class ScorerConfig(BaseModel):
    model_name: str = "unsloth/Qwen3-1.7B"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    max_steps: int = 40
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    component: Literal["task_response", "coherence", "lexical", "grammar"] | None = None
    think_weight: float = 0.0
    score_weight: float = 1.0


class SessionConfig(BaseModel):
    session_id: str | None = None
    models: ModelConfig = Field(default_factory=ModelConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    scorer: ScorerConfig = Field(default_factory=ScorerConfig)
    prompt_policies: dict[str, PromptPolicy] = Field(default_factory=dict)
    strategy_guidance: StrategyGuidance = Field(default_factory=StrategyGuidance)

    def model_post_init(self, __context: Any) -> None:
        from edsmith.data.parser import COMPONENT_HEADINGS
        for key in COMPONENT_HEADINGS:
            self.prompt_policies.setdefault(key, PromptPolicy())

    @classmethod
    def from_yaml(cls, path: Path | str) -> SessionConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: Path | str) -> None:
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
