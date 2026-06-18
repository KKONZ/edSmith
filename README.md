# edSmith

Multi-agent generative feedback system for IELTS writing assessment. A council of LLM agents generates per-component feedback; a lightweight scorer (Qwen3 + LoRA) is fine-tuned on that feedback each iteration; a reflection agent updates the prompt policies based on validation performance.

## Quick start

**1. Install**

```bash
pip install -e ".[dev]"
```

**2. Set your OpenRouter API key**

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

**3. Generate a config**

```bash
edsmith init-config session.yaml
```

**4. Start the Colab training server**

Open a Colab notebook with a GPU runtime and run these two cells:

*Cell 1 — install and start the tunnel:*
```python
!pip install -q unsloth
!pip install -q "edsmith[training] @ git+https://github.com/kkonz/edSmith.git" fastmcp
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
     -O cloudflared && chmod +x cloudflared

import subprocess, re

proc = subprocess.Popen(
    ["./cloudflared", "tunnel", "--url", "http://localhost:8000"],
    stderr=subprocess.PIPE,
    text=True,
)

for line in proc.stderr:
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
    if m:
        print(f"\nMCP URL: {m.group()}/mcp\n")
        break
```

*Cell 2 — start the server (blocking):*
```python
import unsloth  # must be imported before transformers is loaded anywhere
import threading
from edsmith.mcp.server import mcp

t = threading.Thread(target=mcp.run, kwargs={"transport": "http", "port": 8000}, daemon=True)
t.start()
t.join()
```

Copy the printed MCP URL. The URL is the only access control — keep it private while the session is running.

**5. Run the baseline (Session 0) — once only**

Parses the original IELTS dataset into per-component scores and trains the Scorer on that data. Must be run before the first `run-session`.

```bash
edsmith run-baseline --config session.yaml --mcp-url https://<id>.trycloudflare.com/mcp
```

**6. Run a session**

```bash
edsmith run-session --config session.yaml --mcp-url https://<id>.trycloudflare.com/mcp
```

The IELTS dataset downloads automatically from Hugging Face on first run.

---

## Architecture

Each session runs a loop on your **local machine**, calling out to the **Colab GPU server** only for training and evaluation:

```
Local                               Colab (GPU via MCP)
─────────────────────────────────   ───────────────────
Phase 1: generate feedback  ──────────────────────────→ (LLM API calls, no GPU)
Train scorer                ──────────────────────────→ train_scorer tool
Evaluate (val + test)       ──────────────────────────→ evaluate_scorer tool
Reflect: update policies    ──────────────────────────→ (LLM API call, no GPU)
↑___________________________________|
         repeat N iterations
```

---

## Config reference

All options live in `session.yaml`. The key sections:

### `sampling`

Controls how much data is used. All three ratios use `train_full` (after the `size` cap) as their base, so setting the same value for `validation_ratio` and `test_ratio` produces equally-sized splits.

```yaml
sampling:
  size: 500             # cap on the training pool; null = full dataset
                        # if size exceeds the actual dataset it has no effect
  validation_ratio: 0.15   # fraction of the training pool held out for validation
  test_ratio: 0.15         # fraction of the training pool used for test (drawn from the HF test split)
                            # null = use all available test data
  random_state: 42
```

With `size: 500` and both ratios at `0.15`:
- `train_df` ≈ 425 rows → Phase 1 generates feedback, scorer trains on this
- `val_df` ≈ 75 rows → evaluated each iteration
- `test_df` ≈ 75 rows → evaluated each iteration

The CLI prints `Dataset train=... val=... test=...` before any training starts so you can confirm.

### `scorer`

```yaml
scorer:
  model_name: unsloth/Qwen3-1.7B
  component: task_response   # focus training and evaluation on one component
                              # options: task_response | coherence | lexical | grammar
                              # null = all four components (averages predictions for band score)
  max_steps: 40
  lora_r: 16
  lora_alpha: 16
  learning_rate: 0.0002
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  max_seq_length: 4096
  load_in_4bit: true
```

When `component` is set, training filters to only that component's feedback rows (roughly ¼ of the data) and evaluation generates predictions for that component only. Run separate sessions per component to train all four.

### `models`

```yaml
models:
  generator: qwen/qwen3.5-9b   # feedback generation
  critic: qwen/qwen3.5-9b      # council mode only
  chair: anthropic/claude-sonnet-4-5   # council mode + reflection
```

### `council`

```yaml
council:
  enabled: false        # true = Generator → Critic → Chair pipeline
  critic_rounds: 1
  chair_memory_injection: false
```

### `prompt_policies`

One entry per IELTS component. The reflection agent updates these each iteration.

```yaml
prompt_policies:
  task_response:
    specificity: 2        # 1 (brief) to 5 (comprehensive)
    evidence_required: true
    feedback_granularity: component
    additional_instructions: ""
  coherence:   { ... }
  lexical:     { ... }
  grammar:     { ... }
```

---

## Other commands

```bash
# Run tests (no API key needed)
pytest

# Inspect the session history tree
edsmith show-tree

# Write a fresh default config
edsmith init-config session.yaml
```
