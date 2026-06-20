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
# show-tree
# ------------------------------------------------------------------

@app.command("show-tree")
def show_tree(
    drive: Optional[Path] = typer.Option(None, "--drive", help="Path to edsmith drive directory"),
) -> None:
    """Print the episodic memory session tree."""
    from edsmith.memory.episodic import EpisodicMemory

    drive_path = drive or Path("/content/drive/MyDrive/edsmith")
    mem = EpisodicMemory(drive_path=drive_path)
    records = mem.load_all()

    if not records:
        typer.echo("No sessions found.")
        return

    for r in sorted(records, key=lambda r: (r.tree_depth, r.created_at)):
        indent = "  " * r.tree_depth
        acc = f"{r.best_accuracy:.4f}" if r.best_accuracy is not None else "—"
        iters = len(r.iterations)
        typer.echo(
            f"{indent}{r.session_id}"
            f"  depth={r.tree_depth}"
            f"  parent={r.parent_session_id or 'root'}"
            f"  iters={iters}"
            f"  best_acc={acc}"
            f"  arch={r.architecture}"
        )
