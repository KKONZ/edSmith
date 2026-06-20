# RocketSmith Pattern: src/ Layout, Domain MCP Tools, Claude Code Orchestration

The previous architecture had Python agent classes (`FeedbackAgent`, `CouncilAgent`, `ReflectionAgent`) wrapped in a LangGraph `StateGraph` orchestrator. All session logic lived inside Python; the session loop was opaque to the human operator and could not be inspected or redirected mid-run.

We adopt the RocketSmith repository pattern (github.com/ppak10/RocketSmith): a `src/edsmith/` Python library organised into domain subpackages, each exposing its logic as MCP tools; an `agents/` directory at the repo root containing markdown agent definitions; and Claude Code acting as the session orchestrator by calling those tools directly in conversation.

**Why:** The old architecture had the control flow in the wrong place. A Python orchestrator makes it hard to inspect what the agent is doing, inject human judgement mid-session, or add new capabilities without changing the loop. MCP tools expose each capability as a named, typed interface that Claude Code (or any MCP client) can call on demand. Claude Code as orchestrator means the session loop is visible in the conversation, the human can intervene at any step, and the orchestration logic lives in markdown agent definitions that are easy to read and modify. LangGraph was overkill for a linear four-step loop and added a hard dependency that obscured the control flow.

**src/ layout:** `packages = ["src/edsmith"]` in hatchling and `pythonpath = ["src"]` in pytest. This matches standard Python packaging practice, avoids import confusion between the installed package and the local source tree, and mirrors the RocketSmith layout exactly.

**Domain subpackages:** Each domain area is a subpackage of `src/edsmith/` with its own `mcp/` subfolder containing MCP tool definitions. The top-level `src/edsmith/mcp/__main__.py` aggregates all domain tools into a single server. Domains: `examiner/` (feedback generation), `chief_examiner/` (diagnostics and reflection), `training/` (Colab scorer), `tools/` (linguistic features), `session/` (on-disk state).

**No Python agent classes:** All logic moves into MCP tool implementations. There are no `*Agent` Python classes. The `agents/` directory at the repo root contains only markdown files describing how Claude Code should use the tools.

**Considered alternatives:** Keeping LangGraph with MCP tool nodes was considered. Rejected because LangGraph's `StateGraph` adds ceremony around what is fundamentally a linear loop; dropping it reduces dependencies and makes the control flow explicit in the Claude Code conversation instead of hidden inside graph edges.
