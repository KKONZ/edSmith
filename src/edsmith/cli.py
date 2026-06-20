from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

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
        help="Override EDSMITH_DRIVE_PATH (default: /content/drive/MyDrive/edsmith)",
    ),
) -> None:
    """Start the edSmith MCP server (all domain tools registered)."""
    import os

    if drive:
        os.environ["EDSMITH_DRIVE_PATH"] = str(drive)

    typer.echo(f"Starting edsmith MCP server on {host}:{port}")

    from edsmith.mcp.__main__ import mcp
    mcp.run(transport="http", host=host, port=port)


