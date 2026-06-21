# ADR Updates and CLI Test: Examiner-as-CLI Decision

Status: ready-for-agent

## Problem Statement

The examiner pass was converted from an MCP tool to a CLI command to avoid MCP HTTP timeout on 30+ minute batch jobs. The ADR documentation was not updated to reflect this change:

- ADR 0002 still states "All logic is implemented as MCP tools registered in the edSmith FastMCP server" — this is now incorrect.
- ADR 0009 still describes `examiner/` as a domain with an `mcp/` subfolder following the `register_*(app: FastMCP)` pattern — this is now incorrect.
- No ADR captures the decision principle: why long-running batch jobs belong as CLI commands rather than MCP tools.

Additionally, the `examiner-pass` CLI command in `cli.py` (drive path resolution, output formatting, `asyncio.run` wiring) has no dedicated test. The underlying `run_examiner_pass` function in `run.py` is well-tested, but the CLI entry point itself is not.

## Solution

1. Update ADR 0002 to replace the blanket "all logic as MCP tools" statement with an accurate description of the split: stateful reasoning steps are MCP tools; long-running batch jobs are CLI commands.
2. Update ADR 0009 to remove `examiner/` from the MCP domain pattern description and note that the examiner pass is a CLI command.
3. Add a new ADR 0016 documenting the principle that drives the MCP vs CLI decision boundary.
4. Add `tests/test_cli.py` with a thin integration test for the `examiner-pass` command using Typer's `CliRunner`.

## Commits

### Commit 1 — Update ADR 0002

Edit `docs/adr/0002-mcp-tools-conversational-agent-orchestration.md`.

Replace the opening sentence "All logic is implemented as MCP tools registered in the edSmith FastMCP server" with an accurate split:

- Stateful reasoning steps (init_session, run_chief_examiner, approve_proposal, reject_proposal, linguistic tools) are MCP tools.
- Long-running batch jobs (the examiner pass) run as CLI commands instead, for the reason documented in ADR 0016.

No other changes to 0002.

### Commit 2 — Update ADR 0009

Edit `docs/adr/0009-src-layout-domain-mcp-tools.md`.

In the "Domain subpackages" section, remove `examiner/` from the list of domains that expose MCP tools. Add a note that `examiner/` exposes its batch pass as a CLI command (`edsmith examiner-pass`) rather than an MCP tool — see ADR 0016 for the reasoning.

No other changes to 0009.

### Commit 3 — Add ADR 0016

Create `docs/adr/0016-batch-jobs-as-cli-commands.md`.

Content should cover:
- **Context:** The session loop has two kinds of steps: short stateful reasoning steps (seconds to a minute) and long-running batch jobs (minutes to hours). MCP HTTP calls are synchronous and can time out.
- **Decision:** Long-running batch jobs that write all output to disk and require no mid-run reasoning run as CLI commands, not MCP tools. Stateful reasoning steps that require the orchestrator's judgment remain MCP tools.
- **Concrete case:** The examiner pass (`edsmith examiner-pass`) processes hundreds of essays concurrently. On a full dataset this takes 30+ minutes. It has no mid-run decision points — it reads state, writes a parquet, and returns a summary. As a CLI command it streams progress to stderr and is naturally re-entrant.
- **Why not MCP:** An MCP HTTP call that takes 30+ minutes will hit infrastructure timeouts (proxies, load balancers, client-side timeouts). A CLI command has no such constraint.
- **Why not split into chunks:** Chunking would require the orchestrator to manage loop state and reassemble results, adding complexity with no architectural benefit. A single CLI invocation is simpler and naturally atomic.
- **Boundary rule:** If a step (a) runs longer than a few minutes, (b) has no mid-run decision points, and (c) writes all output to disk — it is a CLI command. Otherwise it is an MCP tool.
- **Consequences:** CLI commands print live progress to stderr and return a summary dict. They are tested by calling the underlying async function directly (not via the CLI wrapper). The CLI wrapper itself is tested with Typer's CliRunner.

### Commit 4 — Add CLI integration test

Create `tests/test_cli.py`. Test the `examiner-pass` command using `typer.testing.CliRunner`.

The test should:
- Use the same `session_env` fixture pattern from `tests/examiner/test_feedback.py` (tmp_path drive directory + pre-written train parquet + state.json).
- Invoke `app` from `edsmith.cli` via `CliRunner.invoke`.
- Assert exit code 0.
- Assert that key output strings are present (e.g., "Done", the session ID, "Feedback written to").
- Assert that `feedback_iter0.parquet` was written to the expected path.
- Stub the provider by patching `edsmith.examiner.run.OpenRouterProvider` — or restructure the CLI test to pass `--provider` if that flag is added, otherwise use `monkeypatch`.

The test does NOT need to cover `init-config`, `start-server`, or `show-sessions` — those are out of scope for this issue.

## Decision Document

**ADR 0002 change:** The opening sentence currently misrepresents the architecture. The replacement states the split explicitly: MCP for reasoning steps, CLI for batch jobs.

**ADR 0009 change:** The domain subpackage pattern description currently implies all domains expose MCP tools. `examiner/` is the exception. The update names it explicitly and points to 0016.

**New ADR 0016:** Establishes the boundary rule (long-running + no mid-run decisions + writes-to-disk = CLI). This rule applies to any future step with the same profile — not just the examiner.

**CLI test approach:** `typer.testing.CliRunner` invokes the command in-process with full output capture. This tests the wiring (drive path resolution, asyncio.run, output formatting) without needing a subprocess. The provider is monkeypatched to avoid real API calls, following the pattern in `test_feedback.py`.

**No changes to `run.py`, `feedback.py`, or `cli.py` behavior** — this issue is documentation and test-only.

## Testing Decisions

**What makes a good test:** Test observable CLI behavior — exit code, stdout content, files written. Do not test internal implementation details of `run_examiner_pass` (already covered in `test_feedback.py`).

**Module under test:** `edsmith.cli.examiner_pass` (the typer command).

**Prior art:** `tests/examiner/test_feedback.py::session_env` fixture for on-disk session setup; `TestRunExaminerPass` for the provider stub pattern.

**What not to test:** The business logic of feedback generation — that is covered by `TestRunExaminerPass`. The CLI test only needs to confirm the wiring is correct.

## Out of Scope

- Changes to `examiner/run.py` or `examiner/feedback.py` behavior.
- Tests for `init-config`, `start-server`, or `show-sessions` CLI commands.
- Updates to `.scratch/refactor-examiner-architecture/issues/01-examiner-chief-examiner-refactor.md` (historical record, leave as-is).
- Changes to `agents/edsmith.md` or `agents/examiner.md` (already updated to reflect CLI).
- Any change to how the examiner pass is invoked.
