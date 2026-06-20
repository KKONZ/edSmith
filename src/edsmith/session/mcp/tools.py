from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastmcp import FastMCP

from edsmith.config.session import PromptPolicy
from edsmith.data.parser import COMPONENT_HEADINGS
from edsmith.session.state import SessionState, save_state

_DEFAULT_DRIVE = "/content/drive/MyDrive/edsmith"


def _drive_path() -> Path:
    return Path(os.environ.get("EDSMITH_DRIVE_PATH", _DEFAULT_DRIVE))


def register_init_session(app: FastMCP):
    @app.tool(
        title="Initialize Session",
        description=(
            "Create a new session by saving an initial SessionState to disk. "
            "Generates a session_id if not provided. Initialises default PromptPolicy "
            "for all four IELTS components. Returns the session_id and state file path. "
            "Call this once before the first run_examiner_pass."
        ),
    )
    def init_session(
        session_id: str | None = None,
        parent_session_id: str | None = None,
    ) -> dict:
        drive_path = _drive_path()
        state = SessionState(
            session_id=session_id or str(uuid.uuid4()),
            parent_session_id=parent_session_id,
        )
        for key in COMPONENT_HEADINGS:
            state.policies.setdefault(key, PromptPolicy())

        path = save_state(state, drive_path)
        return {
            "session_id": state.session_id,
            "iteration": state.iteration,
            "parent_session_id": state.parent_session_id,
            "state_path": str(path),
        }

    return init_session
