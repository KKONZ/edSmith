# Colab GPU Setup

Scorer training and evaluation runs on a Colab GPU. Claude Code connects to the Colab session via `colab-mcp` — no tunnel required.

## Prerequisites

- Google account with Colab access
- Google Drive with `MyDrive/edsmith/` directory (created automatically on first session)
- `uv` installed locally (`pip install uv` or https://docs.astral.sh/uv/getting-started/installation/)
- `colab-mcp` connected as an MCP server: `uvx git+https://github.com/googlecolab/colab-mcp`

## Step 1 — Open the Notebook

Open `notebooks/edsmith_training.ipynb` in Colab:

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. File → Open notebook → GitHub → paste the repo URL
3. Select `notebooks/edsmith_training.ipynb`

## Step 2 — Set Runtime to GPU

Runtime → Change runtime type → GPU (T4 is sufficient)

> Colab disconnects idle GPU runtimes after ~90 minutes. Session state on Drive is preserved; re-run the setup cell and continue.

## Step 3 — Run the Setup Cell

Run **Cell 1** once per session. It mounts Drive, installs `edsmith[training]`, and sets `EDSMITH_DRIVE_PATH`.

## Step 4 — Let Claude Code Drive

With the notebook open in your browser, Claude Code uses `colab-mcp` to execute the train and evaluate cells. You do not need to run Cells 2 or 3 manually — Claude Code will set the session variables and call `run_cell`.

## Reconnecting After a Disconnect

1. Reopen the notebook in Colab and re-run the setup cell (Cell 1)
2. Tell Claude Code you are reconnected — it will resume from the last completed step

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `run_cell` fails immediately | Setup cell not run | Run Cell 1 first |
| `[training]` import error | Install failed | Check Cell 1 output for pip errors |
| Drive files appear stale | Cached mount | `drive.flush_and_unmount()` then remount |
| GPU memory error during training | Batch size too large | Reduce `per_device_train_batch_size` to 1 in `ScorerConfig` |
| Runtime restarted unexpectedly | Idle timeout or preemption | Re-run Cell 1; session state is safe on Drive |
