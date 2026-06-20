# Colab GPU Server Setup

The edSmith Scorer training and evaluation runs on a Colab GPU. This guide covers setting up the server and connecting it to your local machine. You need to repeat steps 3–6 each time you reconnect after a disconnect.

## Prerequisites

- Google account with Colab access
- Google Drive with `MyDrive/edsmith/` directory (created automatically on first session)
- The edSmith repo cloned locally with `pip install -e ".[dev]"` already run

## Step 1 — Open the Notebook and Set Runtime

1. Open the edSmith Colab notebook in your browser
2. Go to **Runtime → Change runtime type**
3. Select **GPU** (T4 is sufficient; A100 is faster but not required)
4. Click **Save**

> Colab disconnects idle GPU runtimes after ~90 minutes. If you step away, expect to reconnect.

## Step 2 — Mount Google Drive

Run this cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Verify you can see `/content/drive/MyDrive/edsmith/` before continuing. All session state (parquets, `state.json`, proposals, model checkpoints) lives here.

## Step 3 — Install Training Dependencies

```bash
pip install -e ".[training]" --quiet
```

This installs `unsloth`, `transformers`, and `coral-pytorch`. These are GPU-only — do not run this locally.

## Step 4 — Start the MCP Server

```python
from edsmith.training.mcp.server import mcp
mcp.run(transport="streamable-http", port=8000)
```

The server exposes two tools: `train_scorer` and `evaluate_scorer`. Use `streamable-http` transport — `sse` has known compatibility issues with Cloudflare tunnels.

## Step 5 — Start the Cloudflare Tunnel

In a new cell:

```bash
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared
!./cloudflared tunnel --url http://localhost:8000 &
```

Wait for output like:

```
Your quick Tunnel has been created! Visit it at:
https://something-random.trycloudflare.com
```

Copy this URL. **It changes every time you start a new tunnel.**

## Step 6 — Connect from Your Local Machine

Pass the tunnel URL to any edSmith CLI command:

```bash
edsmith run-session --config session.yaml --mcp-url https://something-random.trycloudflare.com
```

## Reconnecting After a Disconnect

Drive state is preserved across disconnects. When you reconnect:

1. Re-run steps 3–5 to get a fresh tunnel URL
2. Do **not** start a new session — find your existing `session_id` in `state.json`
3. Use Claude Code with the `resume-session` skill to determine where to pick up

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Tunnel URL not printed | cloudflared failed to start | Check for port 8000 conflicts; re-run |
| Connection times out | Server still starting | Wait 30s and retry |
| `train_scorer` errors immediately | `[training]` extras not installed | Check install cell output |
| Drive files appear stale | Cached mount | Run `drive.flush_and_unmount()` then remount |
| GPU memory error during training | Batch size too large | Reduce `per_device_train_batch_size` to 1 in `ScorerConfig` |
| Colab runtime restarted unexpectedly | Idle timeout or preemption | Re-run steps 3–5; session state is safe on Drive |
