---
name: resume-session
description: Use when resuming an interrupted session — reads state.json and checks which files exist to determine the exact point to restart from, without repeating completed work or skipping steps.
---

# Resume Session

## Overview

Each step in the session loop writes a file to Drive before the next step begins. Resumption works by checking which files exist for the current iteration and finding the first missing one. Never re-run a step whose output file already exists and is non-empty.

**Core principle:** Read before acting. The files on Drive are the ground truth — not memory of what you think ran last.

## When to Use

- After a Colab disconnect or timeout mid-session
- After a local machine crash or interrupted CLI command
- When asked to continue a session started in a previous conversation
- When `state.json` shows `iteration > 0` and there is no pending proposal

## Session Loop and File Checkpoints

Each iteration produces these files in order:

| Step | File produced | Tool that produces it |
|---|---|---|
| Examiner pass | `feedback_iter{n}.parquet` | `run_examiner_pass` |
| Train | model checkpoint at `state.model_path` | `train_scorer` (Colab) |
| Evaluate | metrics stored in `state.json` (or `metrics_iter{n}.json`) | `evaluate_scorer` (Colab) |
| Chief Examiner | `proposals/iter{n}.json` with `status=pending` | `run_chief_examiner` |
| Human gate | `proposals/iter{n}.json` with `status=approved\|rejected` | `approve_proposal` / `reject_proposal` |

## Steps

### 1. Load Session State

```
state = load_state(drive_path, session_id)
```

If you do not know the `session_id`, list available sessions:

```
ls {drive_path}/sessions/
```

Pick the most recent directory. Read its `state.json` to confirm it is the right session (`state.iteration`, `state.parent_session_id` if branched).

### 2. Determine Current Iteration

`state.iteration` is incremented only after a proposal is approved. So if `state.iteration == 2`, iteration 2 is in progress (or not yet started) and iterations 0–1 are complete.

### 3. Check Files for the Current Iteration

Work through the checkpoint list in order and find the first gap:

```
n = state.iteration

Check: {drive_path}/sessions/{session_id}/feedback_iter{n}.parquet
  → Missing: resume from Examiner pass (run-examiner-pass skill)

Check: state.model_path exists and points to a real checkpoint
  → Missing: resume from training (call train_scorer via Colab MCP)

Check: metrics present in state.json for iteration n
  → Missing: resume from evaluation (call evaluate_scorer via Colab MCP)

Check: {drive_path}/sessions/{session_id}/proposals/iter{n}.json
  → Missing: resume from Chief Examiner (chief-examiner-review skill)

Check: proposals/iter{n}.json has status == "approved" or "rejected"
  → status == "pending": resume from human gate (present proposal to human)
  → status == "rejected": re-run Chief Examiner with stored critique
  → status == "approved": iteration is complete — state.iteration should be n+1
```

### 4. Resume from the Correct Step

Once you know where to resume:

| Resume point | Action |
|---|---|
| Examiner pass | Use `run-examiner-pass` skill |
| Training | Connect to Colab (see `docs/guides/colab-setup.md`) then call `train_scorer` |
| Evaluation | Connect to Colab then call `evaluate_scorer`; run `interpret-metrics` skill |
| Chief Examiner | Use `chief-examiner-review` skill |
| Human gate | Load proposal with `load_proposal(drive_path, session_id, iteration)` and present it |

### 5. Verify Before Resuming

Before calling any tool, confirm:

- The Colab tunnel is live if you need `train_scorer` or `evaluate_scorer` (see `docs/guides/colab-setup.md`)
- `state.policies` and `state.strategy_guidance` are what you expect — a rejected proposal should not have modified them
- The parquet file, if it exists, is non-empty (`> 0 bytes`) — a partial write during a crash can leave a zero-byte file

## Branched Sessions (MCTS Mode)

If the reflection mode is `mcts` or `beam`, there may be multiple session directories. Each has its own `state.json` with a `parent_session_id` pointing to the session it branched from. To find the current active branch:

- Look for sessions with the highest `iteration` count
- Among those, look for any with a `pending` proposal
- If multiple active branches exist, ask the human which to continue

## Red Flags — Stop and Investigate

- `state.json` does not exist — the session was never initialised or Drive was not mounted when it was created
- Parquet file exists but is 0 bytes — treat as missing; re-run the Examiner pass
- `state.model_path` points to a path that does not exist on Drive — the training step may have completed in Colab but the path was written incorrectly; check Colab output logs
- `proposals/iter{n}.json` status is `approved` but `state.iteration` is still `n` — `approve_proposal` may not have updated state; re-run `approve_proposal`

## Quick Reference

```
Resume order:  feedback parquet → model checkpoint → metrics → proposal → human gate

Key files:
  state.json                          always check first
  feedback_iter{n}.parquet            examiner pass output
  proposals/iter{n}.json              chief examiner output + human decision

If unsure: load_state(), then check files in checkpoint order.
Never re-run a step whose output file already exists and is non-empty.
```
