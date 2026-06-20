"""A2A protocol handler stubs for the Examiner and Chief Examiner agents.

Each handler receives an A2A Task dict and returns a Task result dict.
Full HTTP server implementation is out of scope — these stubs define the
dispatch interface so external orchestrators can target these agents via
the standard A2A protocol once the server is wired up.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Type aliases (A2A protocol shapes — not enforced at runtime)
# ---------------------------------------------------------------------------

Task = dict[str, Any]
TaskResult = dict[str, Any]


# ---------------------------------------------------------------------------
# Examiner handlers
# ---------------------------------------------------------------------------

async def handle_run_examiner_pass(task: Task) -> TaskResult:
    """Dispatch a run_examiner_pass A2A task to the examiner domain.

    Expected task["params"]:
        session_id: str
        iteration: int
        concurrency: int = 4
    """
    raise NotImplementedError(
        "A2A handler not yet wired to edsmith MCP server. "
        "Call run_examiner_pass directly via the edsmith MCP tool."
    )


# ---------------------------------------------------------------------------
# Chief Examiner handlers
# ---------------------------------------------------------------------------

async def handle_run_chief_examiner(task: Task) -> TaskResult:
    """Dispatch a run_chief_examiner A2A task to the chief_examiner domain.

    Expected task["params"]:
        session_id: str
        iteration: int
        critique: str | None = None
    """
    raise NotImplementedError(
        "A2A handler not yet wired to edsmith MCP server. "
        "Call run_chief_examiner directly via the edsmith MCP tool."
    )


async def handle_approve_proposal(task: Task) -> TaskResult:
    """Dispatch an approve_proposal A2A task.

    Expected task["params"]:
        session_id: str
        iteration: int
    """
    raise NotImplementedError(
        "A2A handler not yet wired to edsmith MCP server. "
        "Call approve_proposal directly via the edsmith MCP tool."
    )


async def handle_reject_proposal(task: Task) -> TaskResult:
    """Dispatch a reject_proposal A2A task.

    Expected task["params"]:
        session_id: str
        iteration: int
        critique: str
    """
    raise NotImplementedError(
        "A2A handler not yet wired to edsmith MCP server. "
        "Call reject_proposal directly via the edsmith MCP tool."
    )


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

HANDLERS: dict[str, Any] = {
    "run_examiner_pass": handle_run_examiner_pass,
    "run_chief_examiner": handle_run_chief_examiner,
    "approve_proposal": handle_approve_proposal,
    "reject_proposal": handle_reject_proposal,
}


async def dispatch(skill_id: str, task: Task) -> TaskResult:
    """Route an incoming A2A task to the appropriate handler."""
    handler = HANDLERS.get(skill_id)
    if handler is None:
        return {
            "status": "failed",
            "error": f"Unknown skill_id: {skill_id!r}. "
                     f"Valid skills: {list(HANDLERS)}",
        }
    return await handler(task)
