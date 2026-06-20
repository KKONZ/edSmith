# Refactor: Examiner / Chief Examiner Architecture (RocketSmith pattern)

Status: ready-for-human

## Problem Statement

The current architecture has three structural problems:

1. **The feedback generation agent is strategy-blind.** `CouncilAgent` uses a fixed Generator → Critic → Chair pipeline. The only adaptive lever is `PromptPolicy` text fields updated semi-blindly by the reflection agent. The strategy is locked at code time.

2. **The reflection agent is diagnosis-blind.** `ReflectionAgent` receives only aggregate metrics and has to guess what is wrong with the feedback. It cannot inspect individual training examples, audit feedback quality against objective signals, or identify patterns in what the feedback agent is getting wrong.

3. **The architecture is inverted.** Python agent classes drive the logic from within `edsmith/agents/`, and a LangGraph orchestrator wraps them. The right pattern (following RocketSmith) is to expose all logic as MCP tools and let Claude Code act as the orchestrator — calling tools, reasoning about results, and driving the session loop through conversation.

## Solution

Move all agent logic into MCP tools. Remove the Python agent classes and the Python orchestrator entirely. Claude Code becomes the session orchestrator, calling coarse-grained MCP tools in sequence and presenting results to the human at each step.

**Structure (mirrors RocketSmith):**

```
agents/                  ← markdown agent definitions for Claude Code
  examiner.md            ← how to use the examiner tools
  chief_examiner.md      ← how to run diagnostic and reflection
  edsmith.md             ← how to drive a full session loop
edsmith/                 ← Python library, all exposed as MCP tools
  config/
  data/
  memory/
  metrics.py
  mcp/
    server.py            ← local MCP server (session, examiner, chief examiner, tools)
    colab_server.py      ← Colab GPU server (train, evaluate) — renamed from server.py
    client.py            ← Colab MCP client
  tools/                 ← linguistic feature implementations
    grammar.py
    aoa.py
    complexity.py
  providers/
  cli.py                 ← `edsmith start-server` to start local MCP server
```

`edsmith/agents/` is deleted. `orchestrator.py` is deleted. The session loop moves to Claude Code.

**Local MCP server tools (coarse-grained — one per major session step):**
- `initialize_session(config_path, parent_session_id)` → session_id; loads config, creates state on disk
- `run_examiner_pass(session_id, iteration)` → reads policies + strategy from disk, generates feedback for all training essays concurrently, writes parquet, returns summary
- `run_chief_examiner(session_id, iteration)` → reads feedback + metrics + policies, calls linguistic tools, produces `DiagnosticReport` + `HumanReviewProposal`, saves to disk
- `approve_proposal(session_id, iteration)` → applies proposed policies and strategy to session state on disk
- `reject_proposal(session_id, iteration, critique)` → stores critique, marks proposal rejected so Chief Examiner can re-run with the note
- `get_session_status(session_id)` → returns current session state summary for inspection
- `grammar_check(text)` → grammar error count and types
- `aoa_stats(text)` → Age of Acquisition statistics
- `complexity_stats(text)` → sentence complexity metrics

**Colab MCP server tools (unchanged):**
- `train_scorer(feedback_data, scorer_config, output_dir)` → model_path
- `evaluate_scorer(model_path, eval_data)` → y_true, y_pred

**Session state:** All state (policies, strategy guidance, model paths, metrics, proposals) is persisted to disk under `{drive_path}/sessions/{session_id}/`. Each MCP tool reads and writes what it needs. No in-memory orchestrator holds state between calls. This makes each tool call idempotent and the session resumable.

**Human in the loop:** Claude Code shows the `DiagnosticReport` after calling `run_chief_examiner`, asks the human whether to approve or reject, then calls `approve_proposal` or `reject_proposal(critique)` based on the response. No polling mechanism needed — it is just the normal Claude Code conversation.

**Agent-to-agent communication (A2A):** The `run_examiner_pass` tool internally handles all concurrency (asyncio.gather across essays and components). The `run_chief_examiner` tool can internally re-invoke examiner logic for targeted diagnostic essays. A2A agent cards are defined for both so a future HTTP-serving implementation is well-scoped.

## Commits

### Group A — Remove Python orchestrator and agent classes

**Commit 1:** Delete `edsmith/orchestrator.py`. Delete `edsmith/agents/` directory (all subdirectories). Remove `langgraph` from `pyproject.toml`. Confirm `pytest` still passes (tests that tested agent classes will need updating — those come in later commits).

**Commit 2:** Add root-level `agents/` directory with stub markdown files: `edsmith.md`, `examiner.md`, `chief_examiner.md`. Each file has a heading and a one-line placeholder description. No content yet.

**Commit 3:** Rename `edsmith/mcp/server.py` → `edsmith/mcp/colab_server.py`. Update all references (CLI, README, CLAUDE.md). This clarifies that the existing server is Colab-only and makes room for the new local server.

**Commit 4:** Update `CLAUDE.md` architecture section. Reflect new structure: `agents/` at root for Claude Code definitions, `edsmith/` as pure library, local MCP server for session tools.

**Commit 5:** Add `docs/adr/0009-rocketsmith-pattern.md` documenting the inversion: MCP tools + Claude Code orchestration replaces Python agent classes + Python orchestrator. Covers the LangGraph removal in the same decision.

### Group B — Session state on disk

**Commit 6:** Add `edsmith/session/state.py`. `SessionState` is a Pydantic model persisted as `{drive_path}/sessions/{session_id}/state.json`. Fields: `session_id`, `iteration`, `policies: dict[str, PromptPolicy]`, `strategy_guidance: StrategyGuidance`, `model_path: str | None`, `parent_session_id: str | None`. Add `load_state` / `save_state` helpers.

**Commit 7:** Add `StrategyGuidance` and `DiagnosticReport` Pydantic models to `edsmith/config/session.py`. Add `HumanReviewProposal` persisted as `{drive_path}/sessions/{session_id}/proposals/iter{n}.json`. Remove `CouncilConfig` from `SessionConfig`.

**Commit 8:** Add `docs/adr/0010-session-state-on-disk.md` documenting the decision to persist all session state to disk rather than hold it in an in-process orchestrator, enabling resumability and making each MCP tool call independently inspectable.

### Group C — Linguistic feature tools

**Commit 9:** Create `edsmith/tools/` package with `base.py` defining `ToolResult` TypedDict. Add `tools/grammar.py` using `language_tool_python`. Write unit tests against real text strings.

**Commit 10:** Add `tools/complexity.py` using `spacy`. Write unit tests.

**Commit 11:** Add `tools/aoa.py` stub. Interface fully defined; data loading stubbed with `TODO` pointing to Brysbaert et al. (2019) English Lexicon Project norms. Write unit test that stub returns correctly-shaped `ToolResult`.

**Commit 12:** Update `pyproject.toml`: add `language_tool_python` and `spacy` as optional `[tools]` extras. Add `docs/adr/0011-linguistic-feature-tools.md`.

### Group D — Local MCP server

**Commit 13:** Create `edsmith/mcp/server.py` (new local server, distinct from Colab server). Scaffold `FastMCP("edsmith-local")` with no tools yet. Add `edsmith start-server` CLI command that starts it on a configurable port.

**Commit 14:** Add `initialize_session` and `get_session_status` MCP tools. These cover config loading, state file creation, and session introspection.

**Commit 15:** Add `run_examiner_pass` MCP tool. Internally replicates the logic previously in `CouncilAgent` / `FeedbackAgent` using `asyncio.gather` for concurrency across essays and components. Calls linguistic tools if flagged in `StrategyGuidance`. Reads policies and strategy from session state on disk. Writes feedback parquet and returns a summary dict.

**Commit 16:** Add `run_chief_examiner` MCP tool. Replicates the logic previously in `ReflectionAgent`. Reads feedback parquet + metrics + current policies from session state. Calls linguistic tools to audit feedback quality. Produces `DiagnosticReport` + `HumanReviewProposal`, saves both to disk, returns the proposal as JSON.

**Commit 17:** Add `approve_proposal` and `reject_proposal` MCP tools. `approve_proposal` reads the proposal, applies the new policies and strategy to session state. `reject_proposal` stores the critique and marks the proposal rejected so `run_chief_examiner` can be called again with the human's note in context.

**Commit 18:** Add `grammar_check`, `aoa_stats`, `complexity_stats` MCP tools wrapping `edsmith/tools/`. These are also available as standalone tools for inspection during a Claude Code session.

**Commit 19:** Update `agents/examiner.md` with full content: what the Examiner does, which MCP tools to call, what StrategyGuidance fields mean, how to interpret the feedback summary.

**Commit 20:** Update `agents/chief_examiner.md` with full content: how to run diagnostics, how to interpret a DiagnosticReport, when to propose strategy changes vs. prompt policy changes, how to handle a human rejection.

**Commit 21:** Update `agents/edsmith.md` with full content: the complete session loop (initialize → examiner pass → train → evaluate → chief examiner → human gate → repeat), how to connect to the Colab MCP server, how to resume an interrupted session.

**Commit 22:** Delete all tests that tested the now-removed `CouncilAgent`, `FeedbackAgent`, and `ReflectionAgent` Python classes. Add tests for the new MCP tool functions (called directly, not via HTTP): `run_examiner_pass`, `run_chief_examiner`, `approve_proposal` / `reject_proposal`. Use `StubProvider` from `conftest.py`.

**Commit 23:** Add `docs/adr/0012-human-gate-via-conversation.md` documenting the decision: the human checkpoint is the Claude Code conversation itself. No polling mechanism, no separate approval endpoint beyond `approve_proposal` / `reject_proposal`.

### Group E — A2A protocol (design + stub)

**Commit 24:** Create `edsmith/a2a/` package. Add `cards/examiner.json` and `cards/chief_examiner.json` — fully-defined A2A agent card schemas describing each agent's capabilities.

**Commit 25:** Add `edsmith/a2a/handlers.py` stub. Add `docs/adr/0013-a2a-protocol.md`.

### Group F — ADR updates

**Commit 26:** Update `docs/adr/0002-langgraph-orchestration-crewai-council.md` to reflect the full architectural inversion: no Python orchestrator, no CouncilAgent, Claude Code drives the loop via MCP tools.

**Commit 27:** Update `docs/adr/0008-structured-prompt-policy-as-search-interface.md` to reflect that `StrategyGuidance` extends `PromptPolicy` and both are persisted in session state on disk.

## Decision Document

**Repository structure:**
- `agents/` at root — Claude Code agent definitions (markdown), not Python. Mirrors RocketSmith.
- `edsmith/` — pure Python library. No agent classes. All logic exposed as MCP tools.
- `edsmith/agents/` — deleted entirely.
- `edsmith/orchestrator.py` — deleted entirely.

**Key interfaces:**
- `run_examiner_pass(session_id, iteration)` → `ExaminerSummary` (record count, error count, token usage)
- `run_chief_examiner(session_id, iteration)` → `HumanReviewProposal` (as JSON)
- `approve_proposal(session_id, iteration)` → `SessionState` (updated, as JSON)
- `reject_proposal(session_id, iteration, critique)` → confirmation

**Architectural decisions:**
- All agent logic lives in MCP tool implementations, not in Python classes. Claude Code calls tools; Python implements them.
- Session state is persisted to disk between every tool call. No in-memory orchestrator. Any tool call is re-entrant if the session state files exist.
- Diagnostics restricted to training data. Val/test purity preserved per ADR-0006.
- `StrategyGuidance` is stored in session state on disk and passed to `run_examiner_pass` implicitly (tool reads it from state). Chief Examiner updates it via `approve_proposal`.
- The human checkpoint is the Claude Code conversation. `approve_proposal` / `reject_proposal` are explicit tool calls that Claude Code makes based on human response in the conversation.
- A2A is stub-first. Cards fully defined; HTTP serving is a follow-on.
- AoA data source: Brysbaert et al. (2019). Stub interface defined; data loading is a follow-on.

**Removed:**
- `langgraph` dependency
- `CouncilConfig` from `SessionConfig`
- `edsmith/agents/` directory
- `edsmith/orchestrator.py`
- All Python agent class tests (replaced by MCP tool function tests)

**Added to `pyproject.toml`:**
- Remove: `langgraph`
- Add optional `[tools]` extras: `language_tool_python`, `spacy`

## Testing Decisions

**What makes a good test:** Test MCP tool functions directly (called as Python functions, not via HTTP). Test that they read and write session state correctly, return correctly-shaped output, and handle errors gracefully. Do not test internal prompt strings. Do test pure parsing helpers (score extraction, policy parsing, diagnostic report parsing) exhaustively — they are pure functions with no LLM dependency.

**Modules to test:**
- `edsmith/tools/` — real text strings, no LLM; cover happy path and empty-string edge case
- MCP tool functions in `edsmith/mcp/server.py` — via `StubProvider` and temporary session state directories
- Session state round-trip — `save_state` / `load_state`, `HumanReviewProposal` persistence
- `approve_proposal` / `reject_proposal` — state mutation, not HTTP

**Prior art:**
- `tests/conftest.py::StubProvider` — reuse unchanged
- `tests/test_feedback.py` parsing helper pattern — reuse for score extraction, tag extraction
- `tests/test_reflection.py` mock episodic memory pattern — reuse for chief examiner tool tests

## Out of Scope

- Full A2A HTTP server implementation
- Loading actual AoA data (stub interface only)
- Any web UI for human review
- Reusable NLP feature pipeline for other models
- Scorer architecture changes (Qwen3 + LoRA + CORN loss unchanged)
- Data pipeline changes (loader, parser unchanged)
- Episodic / semantic memory changes
- Colab MCP server changes (`edsmith/mcp/colab_server.py` logic unchanged — only renamed)

## Further Notes

The key conceptual shift: previously Python drove the agents. Now Claude Code drives the tools and Python implements them. This is the same inversion RocketSmith makes. The `agents/` markdown files play the same role as RocketSmith's `agents/openrocket.md` — they tell Claude Code how to use the tools, what to do with the results, and how to reason about the session.

Session resumability is a free benefit of disk-persisted state: if a session is interrupted mid-iteration, Claude Code can call `get_session_status(session_id)` to see where it left off and continue from the last completed tool call.

AoA reference: Brysbaert, M., Mandera, P., McCormick, S. F., & Keuleers, E. (2019). Word prevalence norms for 62,000 English lemmas. Behavior Research Methods, 51(2), 467–479.
