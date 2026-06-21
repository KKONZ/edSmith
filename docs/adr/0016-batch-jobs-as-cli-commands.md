# ADR 0016 — Long-running Batch Jobs as CLI Commands

## Status

Accepted

## Context

The session loop contains two distinct kinds of steps:

- **Stateful reasoning steps** — short operations (seconds to a minute) that require the orchestrator's judgment or update shared state: `init_session`, `run_chief_examiner`, `approve_proposal`, `reject_proposal`, linguistic tool calls. These are natural MCP tools.
- **Long-running batch jobs** — operations that run for minutes to hours, write all output to disk, and require no mid-run decision points: the examiner pass processes hundreds of essays concurrently and takes 30+ minutes on a full dataset.

MCP HTTP calls are synchronous. A blocking call that takes 30+ minutes will hit infrastructure timeouts (proxies, load balancers, client-side timeouts) regardless of how the logic is implemented.

## Decision

Long-running batch jobs run as CLI commands, not MCP tools. Stateful reasoning steps remain MCP tools.

**Boundary rule:** If a step (a) runs longer than a few minutes, (b) has no mid-run decision points, and (c) writes all output to disk — it is a CLI command. Otherwise it is an MCP tool.

**Concrete case — examiner pass:** `edsmith examiner-pass <session_id> <iteration>` reads `state.json`, generates per-component Feedback for all training essays concurrently, writes `feedback_iter{N}.parquet`, and prints a summary. It has no mid-run branching. As a CLI command it streams live progress to stderr and is naturally re-entrant.

**Why not chunk the batch into smaller MCP calls:** Chunking would require the orchestrator to manage loop state and reassemble results across multiple tool calls, adding orchestration complexity with no architectural benefit. A single CLI invocation is simpler and naturally atomic.

## Consequences

- CLI commands print live progress to stderr and return a printed summary on completion.
- The underlying async function (`run_examiner_pass` in `examiner/run.py`) accepts an injectable `provider` parameter for clean test injection without monkeypatching.
- The CLI wrapper is tested with Typer's `CliRunner`; the batch logic is tested by calling the async function directly.
- Adding a future long-running step follows the same pattern: implement the logic as a standalone async function, wrap it in a Typer command, test the function directly and the CLI wrapper via `CliRunner`.
