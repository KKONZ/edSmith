from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_DRIVE = Path(__file__).resolve().parents[2] / "edsmith_drive"

app = typer.Typer(name="edsmith", add_completion=False, no_args_is_help=True)

_BANNER = (
    "\n\033[1;36m"
    "███████╗██████╗ ███████╗███╗   ███╗██╗████████╗██╗  ██╗\n"
    "██╔════╝██╔══██╗██╔════╝████╗ ████║██║╚══██╔══╝██║  ██║\n"
    "█████╗  ██║  ██║███████╗██╔████╔██║██║   ██║   ███████║\n"
    "██╔══╝  ██║  ██║╚════██║██║╚██╔╝██║██║   ██║   ██╔══██║\n"
    "███████╗██████╔╝███████║██║ ╚═╝ ██║██║   ██║   ██║  ██║\n"
    "╚══════╝╚═════╝ ╚══════╝╚═╝     ╚═╝╚═╝   ╚═╝   ╚═╝  ╚═╝\n"
    "\033[0m\033[36m"
    "\n  ─────  multi-agent generative feedback council  ─────\n"
    "\033[0m\n"
)


# ------------------------------------------------------------------
# init-config
# ------------------------------------------------------------------

@app.command("init-config")
def init_config(
    output: Path = typer.Argument(Path("session.yaml"), help="Output path for the config file"),
) -> None:
    """Write a default session config YAML to get started."""
    from edsmith.config.session import SessionConfig

    if output.exists():
        overwrite = typer.confirm(f"{output} already exists. Overwrite?", default=False)
        if not overwrite:
            raise typer.Exit()

    SessionConfig().to_yaml(output)
    typer.echo(f"Default config written to {output}")


# ------------------------------------------------------------------
# start-server
# ------------------------------------------------------------------

@app.command("start-server")
def start_server(
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on"),
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind to"),
    drive: Optional[Path] = typer.Option(
        None, "--drive",
        help="Override EDSMITH_DRIVE_PATH",
    ),
) -> None:
    """Start the edSmith MCP server (all domain tools registered)."""
    import os

    if drive:
        os.environ["EDSMITH_DRIVE_PATH"] = str(drive)

    typer.echo(f"Starting edsmith MCP server on {host}:{port}")

    from edsmith.mcp.__main__ import mcp
    mcp.run(transport="http", host=host, port=port)


# ------------------------------------------------------------------
# show-sessions
# ------------------------------------------------------------------

@app.command("show-sessions")
def show_sessions(
    drive: Optional[Path] = typer.Option(
        None, "--drive",
        help="Override EDSMITH_DRIVE_PATH",
    ),
) -> None:
    """List all sessions, their iteration counter, and any pending proposals."""
    import json
    import os

    drive_path = Path(
        drive or os.environ.get("EDSMITH_DRIVE_PATH", _DEFAULT_DRIVE)
    )
    sessions_dir = drive_path / "sessions"

    if not sessions_dir.exists():
        typer.echo(f"No sessions directory at {sessions_dir}")
        raise typer.Exit()

    rows = sorted(sessions_dir.glob("*/state.json"))
    if not rows:
        typer.echo(f"No sessions found under {sessions_dir}")
        raise typer.Exit()

    for state_file in rows:
        try:
            state = json.loads(state_file.read_text())
            session_id = state.get("session_id", "?")
            iteration = state.get("iteration", 0)
            parent = state.get("parent_session_id") or "root"

            proposal_note = ""
            if iteration > 0:
                prop_path = state_file.parent / "proposals" / f"iter{iteration - 1}.json"
                if prop_path.exists():
                    try:
                        status = json.loads(prop_path.read_text()).get("status", "")
                        if status == "pending":
                            proposal_note = "  ⚠ proposal pending"
                    except Exception:
                        pass

            typer.echo(f"  {session_id}  iteration={iteration}  parent={parent}{proposal_note}")
        except Exception as exc:
            typer.echo(f"  {state_file.parent.name}  (error: {exc})")


# ------------------------------------------------------------------
# examiner-pass
# ------------------------------------------------------------------

@app.command("examiner-pass")
def examiner_pass(
    session_id: str = typer.Argument(..., help="Session ID"),
    iteration: int = typer.Argument(..., help="Iteration number (0-based)"),
    concurrency: int = typer.Option(4, "--concurrency", "-c", help="Max concurrent essay API calls"),
    drive: Optional[Path] = typer.Option(
        None, "--drive",
        help="Override EDSMITH_DRIVE_PATH",
    ),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show tqdm bar and logs on stderr"),
) -> None:
    """Generate per-component Feedback for all training essays in one iteration."""
    import asyncio
    import os

    from edsmith.examiner.run import run_examiner_pass

    drive_path = Path(
        drive or os.environ.get("EDSMITH_DRIVE_PATH", _DEFAULT_DRIVE)
    )

    summary = asyncio.run(
        run_examiner_pass(
            session_id=session_id,
            iteration=iteration,
            drive_path=drive_path,
            concurrency=concurrency,
            progress=progress,
        )
    )

    typer.echo(
        f"\nDone — {summary['essays_processed']}/{summary['essays_total']} essays  "
        f"({summary['components_covered']} with all 4 components)"
    )
    for component, dist in summary.get("score_distributions", {}).items():
        typer.echo(f"  {component}: mean={dist['mean']:.2f}  std={dist['std']:.2f}  n={dist['count']}")
    if summary.get("warnings"):
        typer.echo(f"\n{len(summary['warnings'])} warnings (first 5):")
        for w in summary["warnings"][:5]:
            typer.echo(f"  {w}")
    typer.echo(f"\nFeedback written to {summary['parquet_path']}")
