# Refactor: Examiner / Chief Examiner Architecture (RocketSmith pattern)

Status: ready-for-human

## Problem Statement

The current architecture has three structural problems:

1. **The feedback generation agent is strategy-blind.** `CouncilAgent` used a fixed Generator → Critic → Chair pipeline. The only adaptive lever was `PromptPolicy` text fields updated semi-blindly by the reflection agent.

2. **The reflection agent is diagnosis-blind.** `ReflectionAgent` received only aggregate metrics and had to guess what was wrong with the feedback. It could not inspect individual training examples or audit feedback quality against objective signals.

3. **The architecture is inverted.** Python agent classes drove the logic, and a LangGraph orchestrator wrapped them. The right pattern (following RocketSmith) is to expose all logic as MCP tools, organised by domain, and let Claude Code act as the session orchestrator.

## Solution

Adopt the RocketSmith repository structure:

- `src/edsmith/` — Python library in `src/` layout
- Each domain area is its own subpackage with an `mcp/` subfolder containing its tools
- `src/edsmith/mcp/` is the top-level aggregating server (`__main__.py`)
- `agents/` at repo root contains markdown agent definitions for Claude Code

**Target directory structure:**

```
agents/                        ← Claude Code agent definitions (markdown)
  edsmith.md                   ← session orchestrator (like rocketsmith.md)
  examiner.md
  chief_examiner.md
src/edsmith/
  examiner/                    ← domain: feedback generation
    __init__.py
    feedback.py                ← core generation logic
    mcp/
      __init__.py
      tools.py                 ← run_examiner_pass MCP tool
  chief_examiner/              ← domain: diagnostic / reflection
    __init__.py
    diagnostic.py              ← diagnostic + reflection logic
    mcp/
      __init__.py
      tools.py                 ← run_chief_examiner, approve/reject tools
  training/                    ← domain: scorer training (GPU)
    __init__.py
    scorer.py                  ← Qwen3 + LoRA training/eval (unchanged logic)
    mcp/
      __init__.py
      server.py                ← Colab MCP server (was edsmith/mcp/server.py)
  tools/                       ← domain: linguistic features
    __init__.py
    grammar.py
    aoa.py
    complexity.py
    mcp/
      __init__.py
      tools.py                 ← grammar_check, aoa_stats, complexity_stats
  data/                        ← unchanged
  memory/                      ← unchanged
  config/                      ← unchanged
  session/                     ← new: on-disk session state
    __init__.py
    state.py
  metrics.py                   ← unchanged
  providers/                   ← unchanged
  mcp/
    __init__.py
    __main__.py                ← top-level server; registers all domain tools
    client.py                  ← Colab MCP client (unchanged)
  cli.py
```

**Session state on disk:** All state (policies, strategy guidance, model paths, metrics, proposals) is persisted under `{drive_path}/sessions/{session_id}/`. Each MCP tool reads and writes what it needs. No in-memory orchestrator holds state between calls. Sessions are resumable.

**Human gate:** Claude Code shows the `DiagnosticReport` after calling `run_chief_examiner`, asks the human, then calls `approve_proposal` or `reject_proposal(critique)`. No polling — it is the normal Claude Code conversation.

## Commits

### Status
- ✅ **Commit 1** — Delete `edsmith/agents/`, `orchestrator.py`, LangGraph; strip CLI; delete agent tests
- ✅ **Commit 2** — Add root `agents/` with stub `edsmith.md`, `examiner.md`, `chief_examiner.md`

### Group B — Move to src/ layout and domain structure

**Commit 3:** Unstage partial rename from interrupted Commit 3 attempt. Move entire `edsmith/` → `src/edsmith/` using `git mv`. Update `pyproject.toml`: change `packages = ["edsmith"]` to `packages = ["src/edsmith"]` (hatchling src layout). Update `[tool.pytest.ini_options]` to add `pythonpath = ["src"]` so tests resolve the package. Confirm `pytest` passes with same baseline (2 pre-existing failures, 77 passing).

**Commit 4:** Create domain subpackages inside `src/edsmith/`. Move Colab server: `src/edsmith/mcp/server.py` → `src/edsmith/training/mcp/server.py`. Update the Colab setup docstring import reference. Add empty `__init__.py` files for: `src/edsmith/examiner/`, `src/edsmith/examiner/mcp/`, `src/edsmith/chief_examiner/`, `src/edsmith/chief_examiner/mcp/`, `src/edsmith/training/`, `src/edsmith/tools/`, `src/edsmith/tools/mcp/`, `src/edsmith/session/`. Add `src/edsmith/mcp/__main__.py` stub (empty FastMCP server named `edsmith`). Confirm `pytest` passes.

**Commit 5:** Add `docs/adr/0009-rocketsmith-pattern.md` — documents the architectural inversion (MCP tools + Claude Code orchestration), the `src/` layout choice, the domain subpackage pattern, and the removal of LangGraph and Python agent classes.

### Group C — Session state on disk

**Commit 6:** Add `src/edsmith/session/state.py`. `SessionState` is a Pydantic model with fields: `session_id`, `iteration`, `policies: dict[str, PromptPolicy]`, `strategy_guidance: StrategyGuidance`, `model_path: str | None`, `parent_session_id: str | None`. Add `load_state` / `save_state` helpers persisting to `{drive_path}/sessions/{session_id}/state.json`.

**Commit 7:** Add `StrategyGuidance` and `DiagnosticReport` Pydantic models to `src/edsmith/config/session.py`. Add `HumanReviewProposal` persisted as `{drive_path}/sessions/{session_id}/proposals/iter{n}.json`. Remove `CouncilConfig` from `SessionConfig`. Add `human_in_the_loop: bool = False` field (no-op for now; used later by the MCP server to decide whether to auto-approve).

**Commit 8:** Add `docs/adr/0010-session-state-on-disk.md` and `docs/adr/0011-diagnostic-on-training-data.md`.

### Group D — Linguistic feature tools

**Commit 9:** Add `src/edsmith/tools/grammar.py` using `language_tool_python`. `ToolResult` TypedDict in `src/edsmith/tools/__init__.py`. Write unit tests in `tests/tools/test_grammar.py` against real text strings.

**Commit 10:** Add `src/edsmith/tools/complexity.py` using `spacy`. Write unit tests in `tests/tools/test_complexity.py`.

**Commit 11:** Add `src/edsmith/tools/aoa.py` stub. Interface: `compute_aoa_stats(text: str) -> ToolResult`. Data loading stubbed with `TODO` pointing to Brysbaert et al. (2019). Write `tests/tools/test_aoa.py` verifying the stub returns a correctly-shaped `ToolResult`.

**Commit 12:** Add `src/edsmith/tools/mcp/tools.py` — MCP tools `grammar_check`, `aoa_stats`, `complexity_stats` wrapping the domain modules. Register them in `src/edsmith/mcp/__main__.py`. Update `pyproject.toml` with `language_tool_python` and `spacy` as optional `[tools]` extras. Add `docs/adr/0012-linguistic-feature-tools.md`.

### Group E — Examiner domain

**Commit 13:** Add `src/edsmith/examiner/feedback.py`. Contains the core feedback generation logic previously split across `CouncilAgent` and `FeedbackAgent` — simplified to a single tool-augmented LLM call per component. Uses `asyncio.gather` for concurrency across components. Accepts `StrategyGuidance` and calls linguistic tools when flagged. Preserves `ComponentFeedback` output shape and `_extract_score` / `_extract_tag` parsing helpers.

**Commit 14:** Add `src/edsmith/examiner/mcp/tools.py`. `run_examiner_pass(session_id, iteration)` reads `SessionState` from disk, loads the training DataFrame, generates feedback for all essays concurrently, writes the feedback parquet, and returns a summary dict. Register in `src/edsmith/mcp/__main__.py`.

**Commit 15:** Add `tests/examiner/test_feedback.py`. Test via `StubProvider`: output has correct four components; `StrategyGuidance` is accepted; parsing helpers handle edge cases. Test `run_examiner_pass` tool function directly (not via HTTP) with a temporary session directory.

### Group F — Chief Examiner domain

**Commit 16:** Add `src/edsmith/chief_examiner/diagnostic.py`. Contains diagnostic and reflection logic previously in `ReflectionAgent` — upgraded to receive the training feedback DataFrame, call linguistic tools to audit feedback quality, and produce a `DiagnosticReport` before proposing `StrategyGuidance` updates. Preserves MCTS/beam/simple mode selection and UCB1 scoring.

**Commit 17:** Add `src/edsmith/chief_examiner/mcp/tools.py`. Three tools: `run_chief_examiner(session_id, iteration)` reads feedback parquet + metrics + policies, produces `DiagnosticReport` + `HumanReviewProposal`, saves both to disk, returns proposal JSON. `approve_proposal(session_id, iteration)` applies proposed policies and strategy to `SessionState`. `reject_proposal(session_id, iteration, critique)` stores critique and marks proposal rejected. Register all three in `src/edsmith/mcp/__main__.py`.

**Commit 18:** Add `tests/chief_examiner/test_diagnostic.py`. Test via `StubProvider` + small mock feedback DataFrame: `DiagnosticReport` correctly parsed; policy updates merge correctly; unknown fields ignored; UCB1 mode selection logic unchanged.

### Group G — CLI and top-level server

**Commit 19:** Add `edsmith start-server` CLI command to `src/edsmith/cli.py`. Starts `src/edsmith/mcp/__main__.py` on a configurable port using `fastmcp`. Add `docs/adr/0013-human-gate-via-conversation.md` documenting that the human checkpoint is the Claude Code conversation.

### Group H — Agent markdown content

**Commit 20:** Fill in `agents/examiner.md` with full content: purpose, available MCP tools and their signatures, what `StrategyGuidance` fields mean, how to interpret the feedback summary returned by `run_examiner_pass`.

**Commit 21:** Fill in `agents/chief_examiner.md` with full content: how to interpret a `DiagnosticReport`, when to propose strategy changes vs. prompt policy changes, how to handle a human rejection with critique, how to call `reject_proposal` and re-run.

**Commit 22:** Fill in `agents/edsmith.md` with full orchestration content: the complete session loop step by step, which MCP server each tool lives on, how to connect to the Colab server, how to resume an interrupted session using `get_session_status`.

### Group I — A2A protocol (design + stub)

**Commit 23:** Add `src/edsmith/a2a/` package with `cards/examiner.json` and `cards/chief_examiner.json` — fully-defined A2A agent card schemas. Add stub handlers in `src/edsmith/a2a/handlers.py`. Add `docs/adr/0014-a2a-protocol.md`.

### Group J — ADR updates

**Commit 24:** Update `docs/adr/0002-langgraph-orchestration-crewai-council.md` — reflects full architectural inversion, removal of Python agent classes, removal of LangGraph.

**Commit 25:** Update `docs/adr/0008-structured-prompt-policy-as-search-interface.md` — `StrategyGuidance` extends `PromptPolicy`; both persisted in `SessionState` on disk.

**Commit 26:** Update `CLAUDE.md` — full architecture section rewrite to reflect new `src/` layout, domain subpackages, MCP tool structure, agent markdown files, and session loop.

## Decision Document

**Repository layout mirrors RocketSmith:**
- `src/edsmith/` — Python library in `src/` layout; each domain is a subpackage with its own `mcp/` tools folder
- `agents/` at root — Claude Code agent definitions (markdown); `edsmith.md` is the orchestrator
- No Python agent classes; no Python orchestrator; all logic in MCP tool implementations

**Domain subpackages and their responsibilities:**
- `examiner/` — feedback generation logic; `mcp/tools.py` exposes `run_examiner_pass`
- `chief_examiner/` — diagnostic and reflection logic; `mcp/tools.py` exposes `run_chief_examiner`, `approve_proposal`, `reject_proposal`
- `training/` — Qwen3 + LoRA scorer training and evaluation (unchanged logic); `mcp/server.py` is the Colab server
- `tools/` — linguistic feature implementations; `mcp/tools.py` exposes `grammar_check`, `aoa_stats`, `complexity_stats`
- `session/` — on-disk session state model and helpers
- `mcp/` — top-level aggregating server (`__main__.py`) + Colab client

**Key MCP tool interfaces:**
- `run_examiner_pass(session_id, iteration)` → `ExaminerSummary`
- `run_chief_examiner(session_id, iteration)` → `HumanReviewProposal` (JSON)
- `approve_proposal(session_id, iteration)` → updated `SessionState` (JSON)
- `reject_proposal(session_id, iteration, critique)` → confirmation
- `grammar_check(text)`, `aoa_stats(text)`, `complexity_stats(text)` → `ToolResult`

**pyproject.toml changes:**
- `packages = ["src/edsmith"]` (hatchling src layout)
- Add `pythonpath = ["src"]` to `[tool.pytest.ini_options]`
- Remove `langgraph` (already done in Commit 1)
- Add `[tools]` optional extras: `language_tool_python`, `spacy`

**Session state on disk:** `SessionState` at `{drive_path}/sessions/{session_id}/state.json`. Proposals at `{drive_path}/sessions/{session_id}/proposals/iter{n}.json`. Feedback parquets at `{drive_path}/sessions/{session_id}/feedback_iter{n}.parquet`. Each MCP tool is independently re-entrant.

**Removed:**
- `edsmith/agents/` (Commit 1)
- `edsmith/orchestrator.py` (Commit 1)
- `langgraph` dependency (Commit 1)
- `CouncilConfig` from `SessionConfig` (Commit 7)
- `edsmith/mcp/server.py` → moved to `src/edsmith/training/mcp/server.py` (Commit 4)

## Testing Decisions

**What makes a good test:** Test MCP tool functions directly as Python functions, not via HTTP. Test that they read and write session state correctly and return correctly-shaped output. Test pure parsing helpers exhaustively — they have no LLM dependency.

**Modules to test:**
- `src/edsmith/tools/` — real text strings, no LLM; happy path + empty string
- `src/edsmith/examiner/` — `StubProvider`; correct output shape; `StrategyGuidance` accepted; parsing helpers
- `src/edsmith/chief_examiner/` — `StubProvider` + mock feedback DataFrame; `DiagnosticReport` parsing; policy merge; UCB1 mode selection
- Session state helpers — round-trip `save/load`; proposal status mutation
- MCP tool functions — called directly with temporary session directories

**Prior art:**
- `tests/conftest.py::StubProvider` — reuse unchanged
- Old `test_feedback.py` parsing helper pattern — reuse for score/tag extraction tests
- Old `test_reflection.py` mock memory pattern — reuse for chief examiner tests

## Out of Scope

- Full A2A HTTP server implementation (cards defined, handlers stubbed)
- Loading actual AoA data (stub interface only)
- Any web UI for human review
- Reusable NLP feature pipeline for other models
- Scorer architecture changes (Qwen3 + LoRA + CORN loss unchanged)
- Data pipeline changes (loader, parser unchanged)
- Episodic / semantic memory changes

## Further Notes

The Colab MCP server logic (`training/mcp/server.py`) is unchanged functionally — only its file location moves. The Colab setup instructions in the docstring reference `from edsmith.training.mcp.server import mcp` after the move.

AoA reference: Brysbaert, M., Mandera, P., McCormick, S. F., & Keuleers, E. (2019). Word prevalence norms for 62,000 English lemmas. Behavior Research Methods, 51(2), 467–479.
