# edSmith

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
!pip install -q "edsmith[training]" mcp
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
     -O cloudflared && chmod +x cloudflared

import re, subprocess, threading

def _start_tunnel():
    proc = subprocess.Popen(
        ["./cloudflared", "tunnel", "--url", "http://localhost:8000"],
        stderr=subprocess.PIPE,
    )
    for line in proc.stderr:
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line.decode())
        if m:
            print(f"\nMCP URL: {m.group()}/sse")
            break

threading.Thread(target=_start_tunnel, daemon=True).start()
```

*Cell 2 — start the server (blocking):*
```python
import subprocess
subprocess.run(["python", "-m", "edsmith.mcp.server"])
```

Copy the printed MCP URL. The URL is the only access control — keep it private while the session is running.

**5. Run a session**

Back in your local terminal:

```bash
edsmith run-session --config session.yaml --mcp-url https://<id>.trycloudflare.com/sse
```

The IELTS dataset downloads automatically from Hugging Face on first run.

---

## Config reference

`session.yaml` key fields:

```yaml
n_iterations: 5

models:
  generator: mistralai/mistral-7b-instruct   # feedback generation
  critic: mistralai/mistral-7b-instruct      # council mode only
  chair: anthropic/claude-sonnet-4-5         # council mode + reflection

council:
  enabled: false        # true = Generator→Critic→Chair pipeline
  critic_rounds: 1
  chair_memory_injection: false

sampling:
  size: null            # null = full dataset; e.g. 100 for a quick test

prompt_policies:        # one entry per component; reflection updates these each iteration
  task_response:
    specificity: 2      # 1 (brief) to 5 (comprehensive)
    evidence_required: true
    feedback_granularity: component
    additional_instructions: ""
  # coherence, lexical, grammar follow the same structure
```

---

## Other commands

```bash
# Run tests (no API key needed)
pytest

# Inspect the session history tree
edsmith show-tree
```
