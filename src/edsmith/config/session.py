from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    generator: str = "mistralai/mistral-7b-instruct"
    critic: str = "mistralai/mistral-7b-instruct"
    chair: str = "anthropic/claude-sonnet-4-5"


class PromptPolicy(BaseModel):
    specificity: Annotated[int, Field(ge=1, le=5)] = 2
    evidence_required: bool = True
    feedback_granularity: Literal["component", "overall", "both"] = "component"
    additional_instructions: str = ""


class CouncilConfig(BaseModel):
    enabled: bool = False
    critic_rounds: int = 1
    chair_memory_injection: bool = False


class SamplingConfig(BaseModel):
    size: int | None = None          # None = full dataset; caps train+val+test proportionally
    random_state: int = 42
    validation_ratio: Annotated[float, Field(gt=0, lt=1)] = 0.15
    test_ratio: Annotated[float, Field(gt=0, lt=1)] | None = None  # fraction of test set to use; None = all

    # Stratified reflection sample ratios (must sum to 1.0)
    reflection_correct_ratio: float = 0.30
    reflection_adjacent_ratio: float = 0.40
    reflection_large_miss_ratio: float = 0.30
    reflection_n: int = 30

    @model_validator(mode="after")
    def _ratios_sum_to_one(self) -> SamplingConfig:
        total = (
            self.reflection_correct_ratio
            + self.reflection_adjacent_ratio
            + self.reflection_large_miss_ratio
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Reflection sample ratios must sum to 1.0")
        return self


class ScorerConfig(BaseModel):
    model_name: str = "unsloth/Qwen3-1.7B"
    max_seq_length: int = 512
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    max_steps: int = 40
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    component: Literal["task_response", "coherence", "lexical", "grammar"] | None = None


class MemoryConfig(BaseModel):
    drive_path: str = "/content/drive/MyDrive/edsmith"
    chroma_collection_train: str = "semantic_memory_train"
    chroma_collection_test: str = "semantic_memory_test"


class SessionConfig(BaseModel):
    session_id: str | None = None       # auto-generated if None
    n_iterations: Annotated[int, Field(ge=1)] = 5
    k: Annotated[int, Field(ge=1)] = 4  # few-shot examples retrieved from semantic memory
    phase1_concurrency: Annotated[int, Field(ge=1)] = 4  # max concurrent essays in Phase 1

    models: ModelConfig = Field(default_factory=ModelConfig)
    council: CouncilConfig = Field(default_factory=CouncilConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    scorer: ScorerConfig = Field(default_factory=ScorerConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    prompt_policies: dict[str, PromptPolicy] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _default_prompt_policies(self) -> SessionConfig:
        from edsmith.data.parser import COMPONENT_HEADINGS
        for key in COMPONENT_HEADINGS:
            self.prompt_policies.setdefault(key, PromptPolicy())
        return self

    @classmethod
    def from_yaml(cls, path: Path | str) -> SessionConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: Path | str) -> None:
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
