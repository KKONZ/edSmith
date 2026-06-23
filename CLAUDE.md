# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Domain terminology

Use terms from `CONTEXT.md` exactly. Key ones: **Question** (not prompt/task), **Essay** (not response/submission), **Rubric** (not criteria), **Component** (not criterion/dimension), **Score** (not grade/mark), **Feedback** (not rationale/reasoning), **Scoring Guide** (not rubric when referring to worked examples), **Scorer** (the fine-tuned Qwen3 model), **Prompt Policy** (not prompt template), **Iteration** (not turn/round), **Session** (not run/experiment), **Baseline** (not control/reference model).

## Commands

```bash
uv sync                            # install all dependencies
pytest                             # run all tests (no API key needed)
pytest tests/examiner/             # run a domain test directory
edsmith init-config session.yaml   # write a fresh default config
edsmith start-server --port 8000   # start the edSmith MCP server
edsmith show-sessions              # list active sessions and pending proposals
edsmith examiner-pass <id> <iter>  # run examiner batch pass (CLI, not MCP)
```

`[training]` extras (`unsloth`, `trl`, `transformers`) are GPU-only — install only in the Colab environment, never locally.

## Architecture

Two environments, one shared Drive path:

**Local machine** — runs the edSmith MCP server (`python -m edsmith.mcp`), makes LLM API calls via OpenRouter, reads/writes session state. Claude Code is the session orchestrator, driving the loop via MCP tool calls.

**Colab GPU** — Scorer training and evaluation (Unsloth + Qwen3 + LoRA). The notebook `notebooks/edsmith_training.ipynb` exposes train and evaluate cells. Claude Code drives these cells via `colab-mcp` (`run_cell`) — no tunnel or custom server required.

Both environments read and write to the same `EDSMITH_DRIVE_PATH` (Google Drive mounted in Colab, set as an env var locally).

### Session loop

Claude Code orchestrates each Session by calling edSmith MCP tools in sequence:

```
init_session → run_examiner_pass → [Colab: train → evaluate] → run_chief_examiner → human review → approve_proposal → (next iteration)
```

All state is on disk. Every step is independently resumable. See `agents/edsmith.md` for the full step-by-step guide.

### Source layout (`src/edsmith/`)

Each domain is a subpackage with its own `mcp/tools.py` that registers FastMCP tools via the `register_*(app: FastMCP)` pattern. The top-level server at `src/edsmith/mcp/__main__.py` imports and registers all domain tools.

| Subpackage | Responsibility | MCP tools |
|---|---|---|
| `examiner/` | Per-component Feedback generation | `run_examiner_pass` |
| `chief_examiner/` | Diagnostic and reflection | `run_chief_examiner`, `approve_proposal`, `reject_proposal` |
| `session/` | On-disk session state model and helpers | `init_session` |
| `training/` | Qwen3 + LoRA training and evaluation (GPU-only) | invoked via `run_cell` in Colab notebook |
| `tools/` | Linguistic feature implementations | `grammar_check`, `aoa_stats`, `complexity_stats`, `discourse_analysis` |
| `data/` | IELTS dataset loading and parsing | — |
| `memory/` | Episodic and semantic memory (ChromaDB) | — |
| `config/` | Pydantic config models | — |
| `metrics.py` | Evaluation metrics (deterministic) | — |
| `a2a/` | A2A protocol agent cards and handler stubs | — |

### Examiner (`src/edsmith/examiner/`)

`feedback.py` — `generate_feedback(question, essay, policies, strategy, provider, model_config)` runs all four IELTS components concurrently via `asyncio.gather`. Linguistic tool context (grammar, AoA, complexity, discourse) is collected once via `asyncio.to_thread` before the component calls launch. Returns `{component: ComponentFeedback}`.

`mcp/tools.py` — `run_examiner_pass(session_id, iteration, concurrency)` reads `SessionState`, lazy-initialises session data parquets on first call, runs `generate_feedback` across all training Essays with a semaphore, writes `feedback_iter{N}.parquet`, returns `ExaminerSummary`.

### Chief Examiner (`src/edsmith/chief_examiner/`)

`diagnostic.py` — `run_diagnostic(...)` loads all prior proposals and metrics as an iteration history string, runs a linguistic audit on sampled essays, passes a feedback sample (sorted by largest score-band divergence) to the LLM, and parses the `<diagnostic>` JSON response into a `DiagnosticReport` + `HumanReviewProposal`.

`mcp/tools.py` — three tools: `run_chief_examiner` (produces and saves the proposal), `approve_proposal` (applies proposed policies/strategy to `SessionState`, increments iteration), `reject_proposal` (stores critique, leaves state unchanged).

**Test set purity invariant**: only aggregated test metrics are ever passed to the Chief Examiner — never individual test records. See `docs/adr/0006-*`.

### Session state (`src/edsmith/session/`)

`state.py` — `SessionState` (Pydantic): `session_id`, `iteration`, `policies: dict[str, PromptPolicy]`, `strategy_guidance: StrategyGuidance`, `model_path`, `parent_session_id`. `SessionMetrics`: `val` and `test` metric dicts. `HumanReviewProposal`: diagnostic + proposed changes + status + critique.

All helpers use `{drive_path}/sessions/{session_id}/` as the root:

| File | Written by |
|---|---|
| `state.json` | `init_session`, `approve_proposal` |
| `data/{train,val,test}.parquet` | `run_examiner_pass` (first call) |
| `feedback_iter{N}.parquet` | `run_examiner_pass` |
| `metrics_iter{N}.json` | Colab Cell 3 |
| `models/iter{N}/` | Colab Cell 2 |
| `proposals/iter{N}.json` | `run_chief_examiner`, `approve_proposal`, `reject_proposal` |

### Config (`src/edsmith/config/session.py`)

Key models: `ModelConfig` (generator/critic/chair model IDs), `PromptPolicy` (specificity, evidence_required, feedback_granularity, additional_instructions), `StrategyGuidance` (use_grammar/aoa/complexity/discourse, contrastive_anchoring, per_component_focus), `SessionConfig` (top-level YAML config). `SessionConfig` is used for one-time setup (`init-config`) and `ScorerConfig`; runtime state lives in `SessionState`, not `SessionConfig`.

### Data pipeline (`src/edsmith/data/`)

`loader.py` — downloads IELTS dataset from Hugging Face, splits into train/val/test.
`parser.py` — `parse_evaluation()` splits raw evaluations into per-component sections, extracts Scores. `COMPONENT_HEADINGS` is the canonical mapping: `{"task_response": "Task Achievement", "coherence": "Coherence and Cohesion", "lexical": "Lexical Resource", "grammar": "Grammatical Range and Accuracy"}`.

## Evaluation metrics

All four metrics live in `src/edsmith/metrics.py` and are computed deterministically (no LLM-as-judge — see `docs/adr/0001-*`):
- **accuracy**: exact band match (primary optimization target)
- **adjacent_accuracy**: within one band step (0.5 increments)
- **qwk**: quadratic weighted kappa
- **smd**: standardized mean difference — positive = over-prediction

## Test fixtures

`tests/conftest.py` provides a `StubProvider` fixture that satisfies the `LLMProvider` interface without making API calls. Configure response content via `stub_provider.set(content)`.

## Agent skills

### Issue tracker

Issues live as local markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default label vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
