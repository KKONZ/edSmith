from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from fastmcp import FastMCP

from edsmith.config.session import PromptPolicy, SessionConfig
from edsmith.data.parser import COMPONENT_HEADINGS
from edsmith.session.state import SessionState, save_state

_DEFAULT_DRIVE = Path(__file__).resolve().parents[4] / "edsmith_drive"


def _drive_path() -> Path:
    return Path(os.environ.get("EDSMITH_DRIVE_PATH", _DEFAULT_DRIVE))


def register_init_session(app: FastMCP):
    @app.tool(
        title="Initialize Session",
        description=(
            "Create a new session by saving an initial SessionState to disk. "
            "Generates a session_id if not provided. If config_path is given, "
            "reads models, sampling, and prompt_policies from that session.yaml. "
            "Also saves scorer_config.json to the session directory for Colab Cell 2. "
            "Call this once before the first run_examiner_pass."
        ),
    )
    def init_session(
        session_id: str | None = None,
        parent_session_id: str | None = None,
        config_path: str | None = None,
    ) -> dict:
        drive_path = _drive_path()

        state = SessionState(
            session_id=session_id or str(uuid.uuid4()),
            parent_session_id=parent_session_id,
        )

        scorer_cfg: dict = {}

        if config_path:
            cfg = SessionConfig.from_yaml(config_path)
            state.models = cfg.models
            state.sampling = cfg.sampling
            for key in COMPONENT_HEADINGS:
                state.policies[key] = cfg.prompt_policies.get(key, PromptPolicy())
            scorer_cfg = cfg.scorer.model_dump()
        else:
            for key in COMPONENT_HEADINGS:
                state.policies.setdefault(key, PromptPolicy())

        path = save_state(state, drive_path)

        scorer_path = path.parent / "scorer_config.json"
        scorer_path.write_text(json.dumps(scorer_cfg or {}, indent=2))

        return {
            "session_id": state.session_id,
            "iteration": state.iteration,
            "parent_session_id": state.parent_session_id,
            "state_path": str(path),
            "scorer_config_path": str(scorer_path),
        }

    return init_session
