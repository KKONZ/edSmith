---
name: run-examiner-pass
description: Use when generating Feedback for all training Essays in one iteration. Calls run_examiner_pass, reviews the summary for quality issues, and confirms the feedback parquet was written before proceeding.
---

# Run Examiner Pass

## Overview

The Examiner generates per-component Feedback for every training Essay in parallel. One iteration produces a parquet file at `{drive_path}/sessions/{session_id}/feedback_iter{n}.parquet` containing one row per Essay × Component. The output quality of this pass determines what the Scorer learns — bad Feedback trains a bad Scorer.

**Core principle:** Never mark a pass complete without reviewing the summary. A parquet that exists is not the same as a parquet that is useful.

## When to Use

- Starting iteration N's feedback phase
- Re-running a pass after a rejected `HumanReviewProposal` with critique
- Verifying a pass that completed in a previous (interrupted) session

## Inputs

- `session_id` — from `SessionState.session_id`
- `iteration` — current iteration number from `SessionState.iteration`
- Active `SessionState` at `{drive_path}/sessions/{session_id}/state.json` — must exist before calling

## Steps

### 1. Confirm Session State

Load and review state before calling anything:

```
state = load_state(drive_path, session_id)
```

Check:
- `state.iteration` matches the iteration you intend to run
- `state.policies` has entries for all four components (`task_response`, `coherence`, `lexical`, `grammar`)
- `state.strategy_guidance` is set (or default if iteration 1)
- `state.model_path` is set if iteration > 1 (needed for Scorer evaluation later)

If policies are missing for any component, stop — use defaults from `PromptPolicy()` rather than proceeding with gaps.

### 2. Call run_examiner_pass

```
result = run_examiner_pass(session_id=session_id, iteration=iteration)
```

This call blocks until all Essays are processed. Expect 2–10 minutes depending on dataset size and concurrency setting.

### 3. Review the Summary

The tool returns an `ExaminerSummary` dict. Check each field:

| Field | What to look for |
|---|---|
| `essays_processed` | Should equal training set size. If lower, some Essays failed — check `warnings` |
| `components_covered` | Should be 4 for every Essay. Any < 4 means a parse failure |
| `score_distributions` | Per-component mean and std. Mean should be 5.0–7.0 for IELTS Task 2. Std < 0.3 is suspicious — the model may have collapsed to a single score |
| `warnings` | Any parse failures, timeout errors, or empty Feedback strings |
| `tool_calls_made` | If `StrategyGuidance` flags tools but this is 0, the tools weren't installed — check `[tools]` extras |

### 4. Assess Feedback Quality

Flag problems and re-run if any of these are true:

- **Score collapse** — std < 0.3 on any component across > 50% of essays. The Scorer will learn nothing from homogeneous labels.
- **Low coverage** — more than 5% of Essays are missing any component. The parquet will have gaps that corrupt training.
- **Empty Feedback** — any `ComponentFeedback.feedback` is blank or < 20 characters after stripping. The LLM may have refused or timed out.
- **All scores identical** — mean == median and std == 0. Hard evidence of collapse.

If re-running, consider whether the `PromptPolicy` needs a specificity increase (current level is `state.policies[component].specificity`) or whether `strategy_guidance.per_component_focus` needs updating.

### 5. Confirm Parquet Written

Verify the file exists before proceeding to training:

```
{drive_path}/sessions/{session_id}/feedback_iter{iteration}.parquet
```

If missing, the tool call may have failed silently. Check logs and re-run.

## Red Flags — Stop and Investigate

- `essays_processed` significantly lower than expected dataset size (silent API failures)
- Score mean > 8.0 or < 3.0 on any component across the full training set (systematic prompt bias)
- `warnings` contains LLM provider rate limit errors — wait and retry rather than proceeding with partial output
- Parquet exists from a previous session but `state.iteration` doesn't match the filename — you may be about to overwrite valid data

## Quick Reference

```
Good pass:  essays_processed == train_size, components_covered == 4 per essay,
            score std > 0.5 per component, warnings == []

Rerun if:   score std < 0.3, coverage < 95%, any empty feedback,
            parquet missing after tool returns
```
