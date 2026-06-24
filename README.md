# edSmith

Multi-agent generative feedback system for IELTS writing assessment. An Examiner agent generates per-component Feedback each iteration; a lightweight Scorer (Qwen3 + LoRA) is fine-tuned on that Feedback; a Chief Examiner agent diagnoses quality and proposes updated Prompt Policies and Strategy Guidance for the next iteration. Claude Code orchestrates the loop.

## Quick start

**1. Install**

```bash
uv sync
```

**2. Set environment variables**

```bash
export OPENROUTER_API_KEY="sk-or-..."
export EDSMITH_DRIVE_PATH="/path/to/edsmith"   # local mirror of Google Drive folder
```

**3. Generate a default config**

```bash
edsmith init-config session.yaml
```

**4. Install the Claude Code plugin**

Inside a Claude Code session, add the marketplace and install the plugin at user scope:

```
claude plugin marketplace add KKONZ/edSmith@master
claude plugin marketplace update edsmith
claude plugin install edsmith@edsmith
```

After installing, reload and verify the MCP server is connected:

```
/reload-plugins
/mcp
```

Updating to a new version: uninstall first, then reinstall:

```
/plugin uninstall edsmith@edsmith
/plugin install edsmith@edsmith --scope user
```

**5. Connect a Colab GPU runtime**

Open `notebooks/edsmith_training.ipynb` in a Colab session with a GPU runtime. The notebook is driven via [`SebastianGilPinzon/colab-mcp`](https://github.com/SebastianGilPinzon/colab-mcp) — a community fork that fixes tool visibility and GPU control issues in the official client. Add it to `.claude/settings.json`:

```json
{
  "mcpServers": {
    "colab-proxy-mcp": {
      "type": "stdio",
      "command": "uvx",
      "args": ["git+https://github.com/SebastianGilPinzon/colab-mcp"]
    }
  }
}
```

Make sure Google Drive is mounted in Colab and that `EDSMITH_DRIVE_PATH` points to the same folder in both environments.

**6. Start a session**

Ask Claude Code to start a session. It will follow the guide in `agents/edsmith.md` and call MCP tools in sequence:

```
init_session → run_examiner_pass → [Colab: train → evaluate] → run_chief_examiner → human review → approve_proposal → (next iteration)
```

---

## Architecture

Two environments share a single Google Drive path:

```
Local machine                        Colab GPU
─────────────────────────────────    ──────────────────────────────
edSmith MCP server                   colab-mcp (official plugin)
  init_session                         notebooks/edsmith_training.ipynb
  run_chief_examiner                     Cell 2 — train Scorer
  approve_proposal / reject_proposal     Cell 3 — evaluate, save metrics
  (LLM API calls via OpenRouter)
edsmith examiner-pass (CLI)
  batch feedback generation

        both read/write ──────────────→  EDSMITH_DRIVE_PATH/
                                           sessions/{session_id}/
                                             state.json
                                             data/{train,val,test}.parquet
                                             feedback_iter{N}.parquet
                                             metrics_iter{N}.json
                                             models/iter{N}/
                                             proposals/iter{N}.json
```

Claude Code is the session orchestrator — it calls MCP tools in sequence, handles the human review gate, and drives Colab cells via `run_cell`. All state lives on disk; every step is independently resumable.

---

## CLI

```bash
edsmith init-config [output]                      # write a default session.yaml
edsmith start-server [--port PORT]                # start the edSmith MCP server
edsmith show-sessions [--drive PATH]              # list sessions, iterations, pending proposals
edsmith examiner-pass <session_id> <iteration>    # batch feedback generation (long-running)
```

Run tests (no API key needed):

```bash
pytest
pytest tests/examiner/               # one domain directory
```

---

## Config reference

All options live in `session.yaml`. Runtime state (Prompt Policies, Strategy Guidance, iteration counter) lives in `SessionState` on disk — managed by the MCP tools, not this file.

### `sampling`

```yaml
sampling:
  size: 500             # cap on the training pool; null = full dataset
  validation_ratio: 0.15
  test_ratio: 0.15      # fraction of HF test split to use; null = all
  random_state: 42
```

With `size: 500` and both ratios at `0.15`:
- `train` ≈ 425 rows — Examiner generates Feedback, Scorer trains on this
- `val` ≈ 75 rows — evaluated each iteration, metrics drive Chief Examiner
- `test` ≈ 75 rows — evaluated each iteration, only aggregated metrics reach the Chief Examiner

### `scorer`

```yaml
scorer:
  model_name: unsloth/Qwen3-1.7B
  component: task_response   # one component per session, or null = all four
  max_steps: 40
  lora_r: 16
  lora_alpha: 16
  learning_rate: 0.0002
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  max_seq_length: 512
  load_in_4bit: true
```

When `component` is set, training and evaluation are restricted to that component's rows. Run separate sessions to train all four.

### `models`

```yaml
models:
  generator: mistralai/mistral-7b-instruct   # Examiner feedback generation
  critic: mistralai/mistral-7b-instruct      # reserved
  chair: anthropic/claude-sonnet-4-5         # Chief Examiner diagnostic
```

### `prompt_policies`

One entry per IELTS component. Initial values come from this file; the Chief Examiner proposes updates to these each iteration via `run_chief_examiner` / `approve_proposal`.

```yaml
prompt_policies:
  task_response:
    specificity: 2              # 1 (brief) → 5 (comprehensive)
    evidence_required: true
    feedback_granularity: component   # component | overall | both
    additional_instructions: ""
  coherence:    { ... }
  lexical:      { ... }
  grammar:      { ... }
```

Strategy Guidance (which linguistic tools to inject, contrastive anchoring, per-component focus) is managed by the Chief Examiner at runtime and is not part of `session.yaml`.

---

## Local development

If you're developing the plugin locally and want source changes reflected immediately without reinstalling:

**1. Create `.claude/settings.local.json`** (gitignored — never committed):

```json
{
  "mcpServers": {
    "plugin:edsmith:edsmith": {
      "command": "uv",
      "args": [
        "--directory",
        "C:\\Users\\...\\plugins\\cache\\edsmith\\edsmith\\0.1.0",
        "run",
        "-m",
        "edsmith.mcp"
      ],
      "env": {
        "EDSMITH_DEV_PATH": "C:\\path\\to\\edSmith\\src",
        "EDSMITH_DRIVE_PATH": "C:\\Users\\...\\plugins\\cache\\edsmith\\edsmith\\0.1.0\\edsmith_drive"
      }
    }
  }
}
```

Replace `EDSMITH_DEV_PATH` with the absolute path to your local `src/` directory. The plugin cache path (`--directory` and `EDSMITH_DRIVE_PATH`) can be found by running:

```
claude plugin list
```

**How it works**: `src/edsmith/mcp/__main__.py` checks `EDSMITH_DEV_PATH` at startup and prepends it to `sys.path`, so the local source takes precedence over the installed package. The plugin's own venv still supplies all third-party dependencies.
