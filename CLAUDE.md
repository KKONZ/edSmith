# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Domain terminology

Use terms from `CONTEXT.md` exactly. Key ones: **Question** (not prompt/task), **Essay** (not response/submission), **Rubric** (not criteria), **Component** (not criterion/dimension), **Score** (not grade/mark), **Feedback** (not rationale/reasoning), **Scoring Guide** (not rubric when referring to worked examples), **Scorer** (the fine-tuned Qwen3 model), **Prompt Policy** (not prompt template), **Iteration** (not turn/round), **Session** (not run/experiment), **Baseline** (not control/reference model).

## Commands

```bash
pip install -e ".[dev]"          # install with dev extras (pytest)
pytest                            # run all tests (no API key needed)
pytest tests/test_metrics.py      # run a single test file
edsmith init-config session.yaml  # write a fresh default config
edsmith run-baseline --config session.yaml --mcp-url <url>
edsmith run-session  --config session.yaml --mcp-url <url>
edsmith show-tree                 # inspect episodic memory session tree
```

`[training]` extras (`coral-pytorch`, `transformers`) are GPU-only — install only in the Colab environment, never locally.

## Architecture

The system is split across two environments:

**Local machine** — orchestration, Phase 1 feedback generation (LLM API calls via OpenRouter), reflection, episodic/semantic memory.

**Colab GPU server** — Scorer training and evaluation (Unsloth + Qwen3 + LoRA). Exposed as an MCP server via Cloudflare tunnel. The `--mcp-url` flag connects the local orchestrator to this server via `edsmith/mcp/client.py`. Without `--mcp-url`, the CLI falls back to importing `edsmith.training.scorer` locally (only works if training extras are installed).

### Session loop (`edsmith/orchestrator.py`)

A LangGraph `StateGraph` drives each Session through N iterations:

```
phase1 → train → evaluate → reflect → (loop or END)
```

- **phase1** (`_node_phase1`): Calls `FeedbackAgent` or `CouncilAgent` concurrently (up to `phase1_concurrency`) to generate per-component Feedback for every training Essay. Writes a parquet file per iteration.
- **train** (`_node_train`): Calls `trainer_fn(feedback_df, scorer_config, output_dir)` — runs in a thread executor so async loop doesn't block.
- **evaluate** (`_node_evaluate`): Calls `evaluator_fn(model_path, df)` for validation and test sets; computes accuracy, adjacent_accuracy, QWK, SMD via `edsmith/metrics.py`.
- **reflect** (`_node_reflect`): Calls `ReflectionAgent.areflect()` which suggests `PromptPolicy` updates for the next iteration. Saves `EpisodicRecord` to disk.

DataFrames (train, val, test) are instance attributes on `Orchestrator`, never placed in LangGraph state.

### Phase 1 agents (`edsmith/agents/phase1/`)

- **`FeedbackAgent`** (`feedback.py`): Single-pass LLM call. Generator only.
- **`CouncilAgent`** (`council.py`): Generator → Critic (N rounds) → Chair pipeline. The Chair can optionally receive semantically similar examples from `SemanticMemory` (`chair_memory_injection`). Enabled via `council.enabled` in config.

Both implement `agenerate_all(question, essay, policies)` returning `{component: ComponentFeedback}`.

### Phase 2 reflection (`edsmith/agents/phase2/reflection.py`)

`ReflectionAgent` selects one of three modes based on episodic tree size:
- **simple**: < 5 total nodes and < 2 siblings
- **beam**: ≥ 2 sibling Sessions — compares branches
- **mcts**: ≥ 5 total nodes — uses UCB1 scoring across all Sessions

**Test set purity invariant**: only aggregated summary statistics (not individual records) are ever passed to the reflection agent. Individual record inspection is restricted to the validation split. This is intentional (see `docs/adr/0006-*`).

### Memory (`edsmith/memory/`)

- **`EpisodicMemory`**: Markdown files with YAML frontmatter under `{drive_path}/episodic/`. One file per Session. Tree structure encoded via `parent_session_id` and `tree_depth`. MCTS/UCB1 fields (`visit_count`, `value_estimate`) live here.
- **`SemanticMemory`**: ChromaDB collections for retrieving similar essays during council mode.

### Config (`edsmith/config/session.py`)

All config is `SessionConfig` (Pydantic), loaded from `session.yaml`. Key sub-configs: `ModelConfig`, `PromptPolicy`, `CouncilConfig`, `SamplingConfig`, `ScorerConfig`, `MemoryConfig`. The reflection agent only modifies `PromptPolicy` fields.

### MCP server (`edsmith/mcp/server.py`)

Runs in Colab. Exposes two tools: `train_scorer` (receives base64-encoded parquet, returns model path) and `evaluate_scorer` (returns y_true/y_pred lists). Uses `fastmcp` with streamable-http transport to avoid HTTP/2 issues with Cloudflare tunnels.

### Data pipeline (`edsmith/data/`)

- `loader.py`: Downloads IELTS dataset from Hugging Face, parses raw evaluations via `parser.py`, splits into train/val/test.
- `parser.py`: `parse_evaluation()` splits raw evaluation text into per-component sections and extracts Scores. `COMPONENT_HEADINGS` dict is the canonical mapping from component keys to display names.

## Evaluation metrics

All four metrics live in `edsmith/metrics.py` and are computed deterministically (no LLM-as-judge — see `docs/adr/0001-*`):
- **accuracy**: exact band match
- **adjacent_accuracy**: within one band step (0.5 increments)
- **qwk**: quadratic weighted kappa (primary standard for ordinal scoring)
- **smd**: standardized mean difference — positive = over-prediction

Primary optimization target is **accuracy**.

## Test fixtures

`tests/conftest.py` provides a `StubProvider` fixture that satisfies the `LLMProvider` interface without making API calls. Configure response content via `stub_provider.set(content)`.

## Agent skills

### Issue tracker

Issues live as local markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default label vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
