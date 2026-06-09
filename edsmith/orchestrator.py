from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Callable, TypedDict

import pandas as pd
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from edsmith.agents.phase1.council import CouncilAgent
from edsmith.agents.phase1.feedback import FeedbackAgent
from edsmith.agents.phase2.reflection import ReflectionAgent
from edsmith.config.session import PromptPolicy, SessionConfig
from edsmith.memory.episodic import (
    EpisodicMemory,
    EpisodicRecord,
    IterationMetrics,
    PolicySnapshot,
)
from edsmith.metrics import compute_all


# ------------------------------------------------------------------
# LangGraph state
# ------------------------------------------------------------------

class SessionState(TypedDict):
    iteration: int
    policies: dict             # {component: PromptPolicy.model_dump()}
    iterations: list[dict]     # IterationMetrics.model_dump() per completed iteration
    model_path: str | None     # path to latest trained scorer checkpoint
    feedback_path: str | None  # path to parquet with Phase 1 output
    val_summary: dict          # latest validation metrics
    test_summary: dict | None  # latest test metrics (aggregated only)


# ------------------------------------------------------------------
# Callable protocols (injected from Colab training notebook)
# ------------------------------------------------------------------
#
# trainer_fn(feedback_df, scorer_config, output_dir) -> model_path (str)
# evaluator_fn(model_path, df)                       -> (y_true, y_pred)
#
# These are plain callables so the GPU-dependent Unsloth code stays
# in the Colab notebook and is not imported here.


class Orchestrator:
    """LangGraph-based session loop for edsmith.

    Runs N iterations of:
      Phase 1 — generate per-component feedback for a training sample
      Train   — fine-tune the Scorer on that feedback (via injected trainer_fn)
      Evaluate — score the validation and test sets, compute metrics
      Reflect  — reflection agent suggests PromptPolicy changes

    DataFrames (train, val, test) are held as instance attributes and never
    placed in LangGraph state.  The test set is evaluated here but only its
    aggregated summary statistics are ever passed to the reflection agent.
    """

    def __init__(
        self,
        config: SessionConfig,
        feedback_agent: FeedbackAgent | CouncilAgent,
        reflection_agent: ReflectionAgent,
        episodic_memory: EpisodicMemory,
        trainer_fn: Callable,
        evaluator_fn: Callable,
        drive_path: str | Path,
    ) -> None:
        self._config = config
        self._feedback_agent = feedback_agent
        self._reflection_agent = reflection_agent
        self._episodic = episodic_memory
        self._trainer_fn = trainer_fn
        self._evaluator_fn = evaluator_fn
        self._drive_path = Path(drive_path)

        # Set at run time
        self._train_df: pd.DataFrame | None = None
        self._val_df: pd.DataFrame | None = None
        self._test_df: pd.DataFrame | None = None
        self._record: EpisodicRecord | None = None

        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        parent_session_id: str | None = None,
    ) -> EpisodicRecord:
        return asyncio.run(self.arun(train_df, val_df, test_df, parent_session_id))

    async def arun(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        parent_session_id: str | None = None,
    ) -> EpisodicRecord:
        self._train_df = train_df
        self._val_df = val_df
        self._test_df = test_df
        self._record = self._init_record(parent_session_id)

        initial: SessionState = {
            "iteration": 0,
            "policies": {k: v.model_dump() for k, v in self._config.prompt_policies.items()},
            "iterations": [],
            "model_path": None,
            "feedback_path": None,
            "val_summary": {},
            "test_summary": None,
        }

        thread = {"configurable": {"thread_id": self._record.session_id}}
        await self._graph.ainvoke(initial, config=thread)
        return self._record

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        graph = StateGraph(SessionState)
        graph.add_node("phase1", self._node_phase1)
        graph.add_node("train", self._node_train)
        graph.add_node("evaluate", self._node_evaluate)
        graph.add_node("reflect", self._node_reflect)

        graph.set_entry_point("phase1")
        graph.add_edge("phase1", "train")
        graph.add_edge("train", "evaluate")
        graph.add_edge("evaluate", "reflect")
        graph.add_conditional_edges("reflect", self._route)

        return graph.compile(checkpointer=MemorySaver())

    def _route(self, state: SessionState) -> str:
        return END if state["iteration"] >= self._config.n_iterations else "phase1"

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    async def _node_phase1(self, state: SessionState) -> dict:
        """Generate per-component feedback for the training sample."""
        policies = {k: PromptPolicy(**v) for k, v in state["policies"].items()}
        sample = self._get_sample()

        tasks = [
            self._feedback_agent.agenerate_all(
                question=row["question"],
                essay=row["essay"],
                policies=policies,
            )
            for _, row in sample.iterrows()
        ]
        results = await asyncio.gather(*tasks)

        records = []
        for (_, row), comp_results in zip(sample.iterrows(), results):
            for component, fb in comp_results.items():
                records.append({
                    "question": row["question"],
                    "essay": row["essay"],
                    "band": row["band"],
                    "component": component,
                    "feedback_text": fb.text,
                    "score": fb.score,
                })

        feedback_df = pd.DataFrame(records)
        path = self._drive_path / f"feedback_iter{state['iteration']}.parquet"
        feedback_df.to_parquet(path, index=False)

        return {"feedback_path": str(path)}

    async def _node_train(self, state: SessionState) -> dict:
        """Fine-tune the Scorer on Phase 1 feedback (runs trainer_fn in executor)."""
        feedback_df = pd.read_parquet(state["feedback_path"])
        output_dir = self._drive_path / f"scorer_iter{state['iteration']}"

        loop = asyncio.get_event_loop()
        model_path = await loop.run_in_executor(
            None,
            lambda: self._trainer_fn(feedback_df, self._config.scorer, str(output_dir)),
        )
        return {"model_path": model_path}

    async def _node_evaluate(self, state: SessionState) -> dict:
        """Evaluate the Scorer on val and test sets; update the episodic record."""
        loop = asyncio.get_event_loop()
        model_path = state["model_path"]

        # Validation
        y_true_val, y_pred_val = await loop.run_in_executor(
            None, lambda: self._evaluator_fn(model_path, self._val_df)
        )
        val_metrics = compute_all(y_true_val, y_pred_val)

        # Test — compute full metrics but only surface the summary dict
        y_true_test, y_pred_test = await loop.run_in_executor(
            None, lambda: self._evaluator_fn(model_path, self._test_df)
        )
        test_summary = compute_all(y_true_test, y_pred_test)

        iteration_n = state["iteration"] + 1
        iter_metrics = IterationMetrics(
            iteration=iteration_n,
            accuracy=val_metrics["accuracy"],
            adjacent_accuracy=val_metrics["adjacent_accuracy"],
            qwk=val_metrics["qwk"],
            smd=val_metrics["smd"],
        )

        self._record.iterations.append(iter_metrics)
        self._record.value_estimate = max(
            self._record.value_estimate, val_metrics["accuracy"]
        )

        return {
            "iterations": state["iterations"] + [iter_metrics.model_dump()],
            "val_summary": val_metrics,
            "test_summary": test_summary,
        }

    async def _node_reflect(self, state: SessionState) -> dict:
        """Run reflection agent; update policies and episodic record."""
        current_policies = {k: PromptPolicy(**v) for k, v in state["policies"].items()}

        output = await self._reflection_agent.areflect(
            record=self._record,
            current_policies=current_policies,
            val_summary=state["val_summary"],
            test_summary=state["test_summary"],
        )

        self._record.reflection_notes = output.notes
        self._record.action_taken = output.action_taken
        self._record.visit_count += 1
        self._episodic.save(self._record)

        return {
            "policies": {k: v.model_dump() for k, v in output.suggested_policies.items()},
            "iteration": state["iteration"] + 1,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_sample(self) -> pd.DataFrame:
        cfg = self._config.sampling
        if cfg.size is None:
            return self._train_df
        n = min(cfg.size, len(self._train_df))
        return self._train_df.sample(n=n, random_state=cfg.random_state)

    def _init_record(self, parent_session_id: str | None) -> EpisodicRecord:
        session_id = self._config.session_id or str(uuid.uuid4())[:8]

        depth = 0
        if parent_session_id:
            try:
                parent = self._episodic.load(parent_session_id)
                depth = parent.tree_depth + 1
            except FileNotFoundError:
                pass

        architecture = "council" if self._config.council.enabled else "react"

        return EpisodicRecord(
            session_id=session_id,
            parent_session_id=parent_session_id,
            tree_depth=depth,
            architecture=architecture,
            council_enabled=self._config.council.enabled,
            council_chair_memory_injection=self._config.council.chair_memory_injection,
            prompt_policies={
                k: PolicySnapshot(**v.model_dump())
                for k, v in self._config.prompt_policies.items()
            },
        )
