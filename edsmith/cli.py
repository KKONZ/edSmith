from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

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
# run-session
# ------------------------------------------------------------------

@app.command("run-session")
def run_session(
    config_path: Path = typer.Option(..., "--config", "-c", help="Path to session YAML config"),
    parent: Optional[str] = typer.Option(None, "--parent", "-p", help="Parent session ID (for tree branching)"),
    drive: Optional[Path] = typer.Option(None, "--drive", help="Override drive_path from config"),
    mcp_url: Optional[str] = typer.Option(None, "--mcp-url", help="SSE URL of the Colab MCP training server (e.g. https://<id>.trycloudflare.com/sse)"),
) -> None:
    """Run a full edsmith session (Phase 1 feedback generation → Scorer training → reflection)."""
    from edsmith.agents.phase1.council import CouncilAgent
    from edsmith.agents.phase1.feedback import FeedbackAgent
    from edsmith.agents.phase2.reflection import ReflectionAgent
    from edsmith.config.session import SessionConfig
    from edsmith.data.loader import apply_size_limit, load_with_parsed_evaluations
    from edsmith.data.loader import train_test_split as val_split
    from edsmith.memory.episodic import EpisodicMemory
    from edsmith.memory.semantic import SemanticMemory
    from edsmith.orchestrator import Orchestrator
    from edsmith.providers.openrouter import OpenRouterProvider

    typer.echo(_BANNER)

    # ---- Config ----------------------------------------------------
    if not config_path.exists():
        typer.echo(f"Config not found: {config_path}", err=True)
        raise typer.Exit(code=1)

    cfg = SessionConfig.from_yaml(config_path)
    drive_path = drive or Path(cfg.memory.drive_path)

    typer.echo(f"Session config loaded  (n_iterations={cfg.n_iterations}, council={cfg.council.enabled})")
    typer.echo(f"Local output path: {drive_path.resolve()}")

    # ---- Provider --------------------------------------------------
    try:
        provider = OpenRouterProvider()
    except ValueError as exc:
        typer.echo(f"Provider error: {exc}", err=True)
        raise typer.Exit(code=1)

    # ---- Memory ----------------------------------------------------
    semantic_mem = SemanticMemory(
        drive_path=drive_path,
        collection_train=cfg.memory.chroma_collection_train,
        collection_test=cfg.memory.chroma_collection_test,
    )
    episodic_mem = EpisodicMemory(drive_path=drive_path)

    # ---- Feedback agent --------------------------------------------
    if cfg.council.enabled:
        chair_memory = semantic_mem if cfg.council.chair_memory_injection else None
        feedback_agent: FeedbackAgent | CouncilAgent = CouncilAgent(
            provider=provider,
            generator_model=cfg.models.generator,
            critic_model=cfg.models.critic,
            chair_model=cfg.models.chair,
            critic_rounds=cfg.council.critic_rounds,
            chair_memory=chair_memory,
        )
        typer.echo(f"Council agent  (critic_rounds={cfg.council.critic_rounds}, chair_memory={cfg.council.chair_memory_injection})")
        typer.echo(f"  generator : {cfg.models.generator}")
        typer.echo(f"  critic    : {cfg.models.critic}")
        typer.echo(f"  chair     : {cfg.models.chair}")
    else:
        feedback_agent = FeedbackAgent(
            provider=provider,
            model=cfg.models.generator,
        )
        typer.echo(f"Feedback agent")
        typer.echo(f"  generator : {cfg.models.generator}")
        typer.echo(f"  reflection: {cfg.models.chair}")

    # ---- Reflection agent ------------------------------------------
    reflection_agent = ReflectionAgent(
        provider=provider,
        episodic_memory=episodic_mem,
        model=cfg.models.chair,
    )

    # ---- Dataset ---------------------------------------------------
    typer.echo("Loading dataset …")
    try:
        train_full = load_with_parsed_evaluations(split="train")
        test_df = load_with_parsed_evaluations(split="test")
    except Exception as exc:
        typer.echo(f"Dataset load failed: {exc}", err=True)
        raise typer.Exit(code=1)

    if cfg.sampling.test_ratio is not None:
        if cfg.sampling.size is not None:
            n = min(cfg.sampling.size, len(train_full))
            train_full = train_full.sample(n=n, random_state=cfg.sampling.random_state).reset_index(drop=True)
        # apply test_ratio against train_full size so both ratios use the same base
        n = max(1, min(round(len(train_full) * cfg.sampling.test_ratio), len(test_df)))
        test_df = test_df.sample(n=n, random_state=cfg.sampling.random_state).reset_index(drop=True)
    elif cfg.sampling.size is not None:
        train_full, test_df = apply_size_limit(train_full, test_df, cfg.sampling.size, cfg.sampling.random_state)

    train_df, val_df = val_split(
        train_full,
        validation_ratio=cfg.sampling.validation_ratio,
        random_state=cfg.sampling.random_state,
    )
    typer.echo(f"Dataset  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    # ---- Trainer / evaluator (GPU — via MCP or local fallback) -----
    if mcp_url:
        from edsmith.mcp.client import MCPClient
        _mcp = MCPClient(mcp_url)
        trainer_fn = _mcp.trainer_fn
        evaluator_fn = _mcp.evaluator_fn
        typer.echo(f"MCP training server: {mcp_url}")
    else:
        trainer_fn = _load_trainer()
        evaluator_fn = _load_evaluator()

    # ---- Orchestrator ----------------------------------------------
    orchestrator = Orchestrator(
        config=cfg,
        feedback_agent=feedback_agent,
        reflection_agent=reflection_agent,
        episodic_memory=episodic_mem,
        trainer_fn=trainer_fn,
        evaluator_fn=evaluator_fn,
        drive_path=drive_path,
    )

    # ---- Baseline check -------------------------------------------
    baseline_records = [r for r in episodic_mem.load_all() if r.architecture == "baseline"]
    if not baseline_records:
        typer.echo("WARNING: no baseline (Session 0) found. Run `edsmith run-baseline` first.", err=True)

    typer.echo(f"Starting session …  parent={parent or 'root'}")
    try:
        record = orchestrator.run(train_df, val_df, test_df, parent_session_id=parent)
    except Exception as exc:
        typer.echo(f"Session failed: {exc}", err=True)
        raise typer.Exit(code=1)

    acc = f"{record.best_accuracy:.4f}" if record.best_accuracy is not None else "—"
    typer.echo(f"\nSession {record.session_id} complete  best_accuracy={acc}")
    typer.echo(f"Reflection saved to {drive_path}/episodic/{record.session_id}.md")


# ------------------------------------------------------------------
# run-baseline
# ------------------------------------------------------------------

@app.command("run-baseline")
def run_baseline(
    config_path: Path = typer.Option(..., "--config", "-c", help="Path to session YAML config"),
    drive: Optional[Path] = typer.Option(None, "--drive", help="Override drive_path from config"),
    mcp_url: Optional[str] = typer.Option(None, "--mcp-url", help="SSE URL of the Colab MCP training server"),
) -> None:
    """Train Session 0 (baseline Scorer) on the original parsed dataset.

    Must be run once before the first run-session. Parses the original IELTS
    evaluations into per-component scores and trains the Scorer directly on
    that data, recording the result as the root node in episodic memory.
    """
    from edsmith.config.session import SessionConfig
    from edsmith.data.loader import apply_size_limit, build_baseline_feedback_df, clear_dataset_cache, load_with_parsed_evaluations
    from edsmith.data.loader import train_test_split as val_split
    from edsmith.memory.episodic import EpisodicMemory, EpisodicRecord, IterationMetrics
    from edsmith.metrics import compute_all
    from edsmith.providers.openrouter import OpenRouterProvider

    typer.echo(_BANNER)

    if not config_path.exists():
        typer.echo(f"Config not found: {config_path}", err=True)
        raise typer.Exit(code=1)

    cfg = SessionConfig.from_yaml(config_path)
    drive_path = drive or Path(cfg.memory.drive_path)

    typer.echo(f"Local output path: {drive_path.resolve()}")
    episodic_mem = EpisodicMemory(drive_path=drive_path)
    existing = [r for r in episodic_mem.load_all() if r.architecture == "baseline"]
    if existing:
        typer.echo(f"Baseline already exists (session_id={existing[0].session_id}). Skipping.")
        raise typer.Exit()

    typer.echo("Clearing dataset cache and re-downloading …")
    clear_dataset_cache()
    try:
        train_full = load_with_parsed_evaluations(split="train")
        test_df = load_with_parsed_evaluations(split="test")
    except Exception as exc:
        typer.echo(f"Dataset load failed: {exc}", err=True)
        raise typer.Exit(code=1)

    if cfg.sampling.test_ratio is not None:
        if cfg.sampling.size is not None:
            n = min(cfg.sampling.size, len(train_full))
            train_full = train_full.sample(n=n, random_state=cfg.sampling.random_state).reset_index(drop=True)
        n = max(1, min(round(len(train_full) * cfg.sampling.test_ratio), len(test_df)))
        test_df = test_df.sample(n=n, random_state=cfg.sampling.random_state).reset_index(drop=True)
    elif cfg.sampling.size is not None:
        train_full, test_df = apply_size_limit(train_full, test_df, cfg.sampling.size, cfg.sampling.random_state)

    train_df, val_df = val_split(
        train_full,
        validation_ratio=cfg.sampling.validation_ratio,
        random_state=cfg.sampling.random_state,
    )
    typer.echo(f"Dataset  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    typer.echo("Building baseline feedback DataFrame from parsed evaluations …")
    possible = len(train_df) * 4
    feedback_df = build_baseline_feedback_df(train_df)
    skipped = possible - len(feedback_df)
    typer.echo(f"  {len(feedback_df)} of {possible} component rows have parseable scores  ({skipped} skipped)")

    if mcp_url:
        from edsmith.mcp.client import MCPClient
        _mcp = MCPClient(mcp_url)
        trainer_fn = _mcp.trainer_fn
        evaluator_fn = _mcp.evaluator_fn
        typer.echo(f"MCP training server: {mcp_url}")
    else:
        trainer_fn = _load_trainer()
        evaluator_fn = _load_evaluator()

    output_dir = drive_path / "scorer_baseline"
    typer.echo("Training baseline Scorer …")
    try:
        model_path = trainer_fn(feedback_df, cfg.scorer, str(output_dir))
    except Exception as exc:
        typer.echo(f"Training failed: {exc}", err=True)
        if hasattr(exc, "exceptions"):
            for sub in exc.exceptions:
                typer.echo(f"  caused by: {sub}", err=True)
        raise typer.Exit(code=1)

    typer.echo("Evaluating …")
    try:
        y_true_val, y_pred_val = evaluator_fn(model_path, val_df)
        val_metrics = compute_all(y_true_val, y_pred_val)
        y_true_test, y_pred_test = evaluator_fn(model_path, test_df)
        test_metrics = compute_all(y_true_test, y_pred_test)
    except Exception as exc:
        typer.echo(f"Evaluation failed: {exc}", err=True)
        if hasattr(exc, "exceptions"):
            for sub in exc.exceptions:
                typer.echo(f"  caused by: {sub}", err=True)
        raise typer.Exit(code=1)

    record = EpisodicRecord(
        session_id="baseline",
        architecture="baseline",
        tree_depth=0,
        reflection_notes=(
            "Session 0: baseline Scorer trained on original parsed IELTS evaluations. "
            f"Val accuracy={val_metrics['accuracy']:.4f}  "
            f"Test accuracy={test_metrics['accuracy']:.4f}"
        ),
        iterations=[IterationMetrics(
            iteration=0,
            accuracy=val_metrics["accuracy"],
            adjacent_accuracy=val_metrics["adjacent_accuracy"],
            qwk=val_metrics["qwk"],
            smd=val_metrics["smd"],
        )],
    )
    episodic_mem.save(record)

    typer.echo(
        f"\nBaseline complete"
        f"  val_acc={val_metrics['accuracy']:.4f}"
        f"  test_acc={test_metrics['accuracy']:.4f}"
        f"  qwk={val_metrics['qwk']:.4f}"
    )
    typer.echo(f"Saved to {drive_path}/episodic/baseline.md")


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


# ------------------------------------------------------------------
# Trainer / evaluator loaders
# ------------------------------------------------------------------

def _load_trainer() -> Callable:
    try:
        from edsmith.training.scorer import train_scorer
        return train_scorer
    except ImportError:
        def _stub(feedback_df, scorer_config, output_dir):
            raise RuntimeError(
                "edsmith.training.scorer is not available. "
                "Scorer training requires a GPU environment with Unsloth installed. "
                "Run from within colab/scorer_training.ipynb, or install the training extras."
            )
        return _stub


def _load_evaluator() -> Callable:
    try:
        from edsmith.training.scorer import evaluate_scorer
        return evaluate_scorer
    except ImportError:
        def _stub(model_path, df):
            raise RuntimeError(
                "edsmith.training.scorer is not available. "
                "Scorer evaluation requires a GPU environment with Unsloth installed."
            )
        return _stub
