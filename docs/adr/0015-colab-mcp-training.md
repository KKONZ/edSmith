# ADR 0015 — Colab GPU via colab-mcp

## Status

Accepted

## Context

Scorer training and evaluation (Qwen3 + LoRA + CORN loss) requires a Colab GPU runtime. Claude Code needs a way to trigger training and evaluation cells and retrieve results without manual URL management.

## Decision

Use `googlecolab/colab-mcp`. It runs locally and bridges to an open Colab browser session, exposing `run_cell` and `add_cell` tools. Claude Code calls `run_cell` to execute cells in `notebooks/edsmith_training.ipynb` on the Colab GPU.

Training and evaluation logic lives in `training/scorer.py`, exposing `train(session_id, iteration, drive_path)` and `evaluate(session_id, iteration, split, drive_path)`. Notebook cells call these functions; Claude Code prepends variable assignment (`SESSION_ID`, `ITERATION`, `SPLIT`) before calling `run_cell`.

`colab-mcp` is installed via `uvx git+https://github.com/googlecolab/colab-mcp` and connected as an MCP server when running a session.

## Consequences

- The Colab notebook must be open in the browser during training — colab-mcp bridges to a live browser session.
- `run_cell` latency is negligible relative to multi-minute training jobs.
- No persistent server process to manage between sessions.
