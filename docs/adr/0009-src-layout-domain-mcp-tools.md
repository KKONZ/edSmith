# src/ Layout, Domain Subpackages, and MCP Tools

The edSmith codebase uses a `src/edsmith/` Python library organised into domain subpackages. Each domain subpackage exposes its logic as MCP tools registered on a shared FastMCP server. An `agents/` directory at the repo root contains markdown agent definitions. The session orchestrator (any MCP-compatible AI agent) calls those tools directly in conversation.

**src/ layout:** `packages = ["src/edsmith"]` in hatchling and `pythonpath = ["src"]` in pytest. This matches standard Python packaging practice and avoids import confusion between the installed package and the local source tree.

**Domain subpackages:** Each domain area is a subpackage of `src/edsmith/` with its own `mcp/` subfolder containing a `register_*(app: FastMCP)` function. The top-level `src/edsmith/mcp/__main__.py` creates the single `FastMCP` instance and calls each `register_*` function. Domains with MCP tools: `chief_examiner/` (diagnostics), `tools/` (linguistic features), `session/` (on-disk state). The `examiner/` domain exposes its batch pass as the `edsmith examiner-pass` CLI command rather than an MCP tool — see ADR 0016.

**No Python agent classes:** All domain logic lives in MCP tool implementations. There are no `*Agent` Python classes. The `agents/` directory contains only markdown files describing how the session orchestrator should use the tools.

**Why MCP tools:** MCP tools expose each capability as a named, typed interface that any MCP-compatible orchestrator can call on demand. The session loop is visible in the conversation, the human can intervene at any step, and the orchestration logic lives in markdown files that are easy to read and modify.

**Considered alternative — graph framework with MCP tool nodes:** A `StateGraph` adds ceremony around what is fundamentally a linear loop, requires framework-specific serialisation, and makes the control flow implicit inside graph edges rather than explicit in the conversation. Dropped in favour of direct MCP tool calls.
