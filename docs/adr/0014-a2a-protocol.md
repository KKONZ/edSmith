# A2A Protocol — Agent Cards and Handler Stubs

The Examiner and Chief Examiner are defined as A2A agents with fully-specified agent cards at `src/edsmith/a2a/cards/`. Dispatch handlers are stubbed in `src/edsmith/a2a/handlers.py`. The A2A HTTP server layer is out of scope for the initial implementation.

**Why:** The Claude Code plugin and MCP server are the primary runtime for edSmith sessions, but the Examiner and Chief Examiner are independently useful agents — they have well-defined input/output contracts, stateless APIs (state lives on disk), and no dependency on Claude Code as the orchestrator. Expressing them as A2A agents makes them addressable by any A2A-compatible orchestration framework (AWS multi-agent, Kiro, Google ADK) without changing the domain logic. The agent cards are the interface contract; the MCP tools are the implementation.

**What the cards define:** Each card specifies the agent's `url`, `capabilities`, `authentication` scheme, and `skills`. Each skill includes a JSON Schema for `inputSchema` and `outputSchema` so external orchestrators can validate calls without reading source code. The Examiner card defines one skill (`run_examiner_pass`). The Chief Examiner card defines three (`run_chief_examiner`, `approve_proposal`, `reject_proposal`).

**Handler stubs:** `handlers.py` defines one async handler per skill and a `dispatch(skill_id, task)` entry point. All handlers currently raise `NotImplementedError` with a hint to use the MCP tool directly. When the A2A HTTP server is implemented, handlers will call the same domain functions the MCP tools call — no duplication of business logic.

**What is not implemented:** The A2A HTTP server (`POST /a2a/{agent}/tasks/send`, `GET /a2a/{agent}/tasks/{id}`), streaming support, and push notifications. The `capabilities` block in each card reflects this: `streaming: false`, `pushNotifications: false`.

**Considered alternative — expose MCP tools as A2A directly:** MCP and A2A serve different orchestration models. MCP is Claude Code's tool protocol; A2A is for agent-to-agent task delegation across frameworks. Wrapping MCP tools as A2A skills keeps both protocols thin and lets the domain logic stay framework-agnostic.
