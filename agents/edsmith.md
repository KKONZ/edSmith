# edSmith Session Agent

Orchestrates a complete edSmith session: initialize → examiner pass → train scorer → evaluate → chief examiner → human review → repeat until satisfied with Scorer accuracy.

## Purpose

This agent drives the full iteration loop for a Session. It coordinates the Colab GPU training environment and the human reviewer. Each iteration produces a feedback parquet, a trained Scorer checkpoint, evaluation metrics, a diagnostic, and a human-approved set of changes for the next iteration.

State is fully on disk — every step is independently resumable.

## MCP Servers and CLI

Two MCP servers are in scope during a session:

| Server | What it provides | How to start |
|---|---|---|
| **edsmith** | `init_session`, `run_chief_examiner`, `approve_proposal`, `reject_proposal`, and linguistic tools | `edsmith start-server` |
| **colab-mcp** | `run_cell`, `add_cell` — bridges to the Colab browser session for GPU training | `uvx git+https://github.com/googlecolab/colab-mcp` (run locally; Colab notebook must be open) |

The examiner pass runs as a **CLI command** via the terminal tool (not MCP) — it is a long-running batch job that prints live progress:

```
edsmith examiner-pass <session_id> <iteration> [--concurrency 4]
```

The edsmith server tools all read `EDSMITH_DRIVE_PATH` for the session data location. The colab-mcp tools drive the open Colab notebook — no URL or tunnel configuration needed.

## Environment Check (SessionStart)

At the start of every conversation the `SessionStart` hook runs and injects a status block into context:

```
OPENROUTER_API_KEY: set | NOT SET
EDSMITH_DRIVE_PATH: set (/path/to/data) | using default
spacy [tools]: installed | NOT INSTALLED
language_tool_python [tools]: installed | NOT INSTALLED
status: ready | NOT READY — missing: ...

Active sessions:
  <session_id>  iteration=N  parent=root  [⚠ proposal pending]
```

Check this before doing anything. If `OPENROUTER_API_KEY` is not set, LLM calls will fail immediately. If a session shows `⚠ proposal pending`, a `HumanReviewProposal` is waiting for `approve_proposal` or `reject_proposal` from the previous conversation.

## Connecting to Colab

Before running any training or evaluation steps:

1. Open `notebooks/edsmith_training.ipynb` in Colab (File → Open → GitHub → paste repo URL).
2. Set the runtime to GPU (Runtime → Change runtime type → T4 GPU).
3. Run **Cell 1** once — mounts Drive, installs `edsmith[training]`, sets `EDSMITH_DRIVE_PATH`.
4. Confirm `colab-mcp` is connected as an MCP server in this session.

After Cell 1 completes, the agent drives all subsequent cell execution via `run_cell`. You do not run Cells 2 or 3 manually.

If the Colab runtime disconnects (idle timeout ~90 min): re-open the notebook, re-run Cell 1, and reconnect. Session data on Drive is safe — resume from the last completed step.

## Complete Session Loop

### Step 0 — Initialize a new session

Call `init_session` once per new session. Pass `config_path` to load model, sampling, and policy settings from `session.yaml`.

```
init_session(session_id=None, parent_session_id=None, config_path="session.yaml")
→ {session_id, iteration, parent_session_id, state_path, scorer_config_path}
```

Save the returned `session_id` — you will pass it to every subsequent step. If branching from a prior session, pass `parent_session_id` to link the sessions in the history tree.

---

### Step 1 — Run Examiner pass

```bash
edsmith examiner-pass <session_id> <iteration> [--concurrency 4]
```

Reads current `StrategyGuidance` and `PromptPolicy` from `state.json`. On the first call, downloads and splits the IELTS dataset into train/val/test parquets under `sessions/{session_id}/data/` using the sampling config from state. Generates per-component Feedback for all training Essays concurrently.

Writes: `sessions/{session_id}/feedback_iter{N}.parquet`

Check the printed summary:
- If `essays_processed` < `essays_total` by more than 5%, check the warning list and re-run before proceeding.
- If `components_covered` < `essays_processed`, note it — partial coverage is diagnosable but worth flagging to the Chief Examiner.

See `agents/examiner.md` for full field reference.

---

### Step 2 — Train Scorer (Colab)

Prepend variable assignments and call `run_cell` on Cell 2 of the notebook:

```python
SESSION_ID = '<session_id>'
ITERATION = <N>
# [Cell 2 body follows — do not modify]
```

Cell 2 reads `feedback_iter{N}.parquet` and `scorer_config.json`, fine-tunes Qwen3 with LoRA, and writes the model to `sessions/{session_id}/models/iter{N}/`. The cell prints `model_path:` on completion.

Training takes 5–15 minutes on a T4 GPU depending on dataset size and `max_steps`. Wait for the cell to complete before proceeding.

---

### Step 3 — Evaluate Scorer (Colab)

Prepend variable assignments and call `run_cell` on Cell 3:

```python
SESSION_ID = '<session_id>'
ITERATION = <N>
# [Cell 3 body follows — do not modify]
```

Cell 3 evaluates the trained model on both the validation and test splits, prints metrics for both, and saves `sessions/{session_id}/metrics_iter{N}.json` to Drive. The saved file contains:

```json
{
  "session_id": "...",
  "iteration": N,
  "val": {"accuracy": 0.48, "adjacent_accuracy": 0.81, "qwk": 0.71, "smd": +0.21},
  "test": {"accuracy": 0.44, "adjacent_accuracy": 0.79, "qwk": 0.68, "smd": +0.18}
}
```

The four metrics:
- **accuracy** — exact band match (primary optimization target)
- **adjacent_accuracy** — within one 0.5-increment band step
- **qwk** — quadratic weighted kappa (standard ordinal agreement measure)
- **smd** — standardized mean difference; positive = Scorer over-predicts

---

### Step 4 — Run Chief Examiner

```
run_chief_examiner(session_id, iteration)
→ {diagnostic_summary, per_component_issues, proposed_strategy, proposed_policies, ...}
```

Reads the feedback parquet and `metrics_iter{N}.json`. Loads all prior proposals and metrics as iteration history. Produces a `DiagnosticReport` and `HumanReviewProposal`, saved to `sessions/{session_id}/proposals/iter{N}.json`.

Returns an error dict (not an exception) if either the feedback parquet or metrics file is missing — check for an `error` key before proceeding.

See `agents/chief_examiner.md` for full diagnostic interpretation, record-level inspection patterns, and the strategy-vs-policy decision guide.

---

### Step 5 — Human review

Show the human:
- `diagnostic_summary` — the Chief Examiner's main finding
- `per_component_issues` — per-component breakdown
- `proposed_strategy` — which linguistic tools to enable, per-component focus instructions
- `proposed_policies` — changes to specificity, evidence, granularity, additional instructions

The human either approves or provides a critique.

**Approve:**
```
approve_proposal(session_id, iteration)
→ {next_iteration, updated_policies, updated_strategy}
```
Applies proposed changes to `state.json` and increments the iteration counter. Proceed to Step 1 with `iteration = next_iteration`.

**Reject with critique:**
```
reject_proposal(session_id, iteration, critique="...")
run_chief_examiner(session_id, iteration, critique="...")
```
Stores the critique, re-runs the diagnostic with the critique included in the LLM prompt. Show the revised proposal to the human and repeat until approved.

---

### Step 6 — Loop or finish

Repeat Steps 1–5 for each iteration. Stop when:
- Val accuracy reaches the target (≥ 0.55 is a reasonable first milestone for IELTS band scoring).
- The human is satisfied with the trend across iterations.
- Diminishing returns are evident from the iteration history in `run_chief_examiner`.

## Resuming an Interrupted Session

Every step writes to disk before returning. If the conversation ends or a step fails:

1. **Check the SessionStart hook output** — it lists all active sessions with their current iteration and any pending proposals.
2. **Identify the last completed step** for the interrupted session:
   - `state.json` exists → session was initialized
   - `feedback_iter{N}.parquet` exists → Step 1 completed for iteration N
   - `models/iter{N}/` exists → Step 2 completed for iteration N
   - `metrics_iter{N}.json` exists → Step 3 completed for iteration N
   - `proposals/iter{N}.json` with `status: "pending"` → Step 4 completed, awaiting human review
3. **Re-run from the next incomplete step.** All steps are re-entrant — re-running a completed step overwrites its output (use this intentionally if you want to re-generate feedback with the same policy).

If a pending proposal is present from a previous conversation, call `approve_proposal` or `reject_proposal` before starting a new examiner pass at the incremented iteration.

## edsmith MCP Tools (quick reference)

| Tool | Step | Description |
|---|---|---|
| `init_session` | 0 | Create `state.json` for a new session |
| `run_chief_examiner` | 4 | Diagnose feedback quality, produce proposal |
| `approve_proposal` | 5 | Apply proposed changes, advance iteration |
| `reject_proposal` | 5 | Store critique, leave state unchanged |
| `grammar_check` | — | Grammar error analysis for a text |
| `aoa_stats` | — | Vocabulary Age-of-Acquisition statistics |
| `complexity_stats` | — | Syntactic complexity metrics |
| `discourse_analysis` | — | Paragraph structure and transition analysis |

Scorer training and evaluation run in Colab via colab-mcp `run_cell`. The examiner pass runs via the `edsmith examiner-pass` CLI command.

## On-disk Layout

```
{EDSMITH_DRIVE_PATH}/sessions/{session_id}/
  state.json                      ← SessionState (policies, strategy, iteration, models, sampling)
  scorer_config.json              ← ScorerConfig snapshot written by init_session
  data/
    train.parquet                 ← IELTS training split (written on first examiner pass)
    val.parquet
    test.parquet
  feedback_iter{N}.parquet        ← Examiner output for iteration N
  metrics_iter{N}.json            ← Val + test metrics for iteration N
  models/iter{N}/                 ← Trained Scorer checkpoint for iteration N
  proposals/iter{N}.json          ← HumanReviewProposal for iteration N
```
