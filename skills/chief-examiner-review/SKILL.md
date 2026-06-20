---
name: chief-examiner-review
description: Use after Scorer evaluation to run the Chief Examiner diagnostic, interpret the DiagnosticReport, present the HumanReviewProposal, and process the human's approve or reject decision.
---

# Chief Examiner Review

## Overview

The Chief Examiner reads the training Feedback parquet, runs linguistic tool audits if flagged, and produces a `DiagnosticReport` + `HumanReviewProposal`. The human then approves or rejects. This is the only mandatory human gate in the session loop — do not auto-approve or skip it.

**Core principle:** Read the `DiagnosticReport` before presenting anything to the human. The proposal makes no sense without understanding what the diagnostic found.

## When to Use

- After `interpret-metrics` has been run and the metric brief is ready
- When starting the reflection phase of an iteration
- When resuming a session that has a `pending` proposal at `proposals/iter{n}.json`

## Inputs

- `session_id`, `iteration` — from `SessionState`
- Metric brief from `interpret-metrics` skill
- Training feedback parquet at `{drive_path}/sessions/{session_id}/feedback_iter{iteration}.parquet` — must exist

## Steps

### 1. Call run_chief_examiner

```
proposal = run_chief_examiner(session_id=session_id, iteration=iteration)
```

This reads the feedback parquet, optionally calls linguistic tools (controlled by `StrategyGuidance`), and writes both `DiagnosticReport` and `HumanReviewProposal` to disk before returning.

Expect 1–5 minutes. The proposal JSON is saved at:
```
{drive_path}/sessions/{session_id}/proposals/iter{iteration}.json
```

### 2. Read the DiagnosticReport

Before showing the human anything, read `proposal.diagnostic_report`:

| Field | What to check |
|---|---|
| `summary` | High-level narrative — does it align with your metric brief? |
| `per_component_issues` | Issues called out per component — are they specific or vague? |
| `metric_summary` | Confirm these match what evaluate_scorer returned |
| `linguistic_findings` | If tools were called: are grammar errors concentrated on low-AoA words? Is the essay structure weak? |

If the diagnostic summary contradicts the metrics (e.g. declares "feedback quality is good" when QWK dropped), the Chief Examiner may have received stale or partial data — do not present the proposal yet. Investigate first.

### 3. Present the Proposal to the Human

Format the proposal clearly. Show the human:

**What changed:**
- `proposed_strategy` — any changes to `use_grammar`, `use_aoa`, `use_complexity`, `use_discourse`, `contrastive_anchoring`, `per_component_focus`
- `proposed_policies` — changes to `specificity`, `evidence_required`, `feedback_granularity`, `additional_instructions` per component

**Why** — pull the most relevant sentences from `diagnostic_report.summary` and `per_component_issues`.

Example presentation:

```
Chief Examiner Proposal — Iteration 3

DIAGNOSTIC FINDINGS:
- Lexical Resource: Feedback is describing vocabulary as "varied" on essays
  with low AoA mean (6.2) and high repetition rate (0.41). Examiner is
  over-rewarding without checking actual lexical sophistication.
- Grammar: 73% of errors flagged on AoA < 5 words — basic errors being
  missed or under-penalised.

PROPOSED CHANGES:
  Strategy:   use_aoa → True, use_grammar → True
              per_component_focus[lexical] → "Focus on AoA distribution
              and type-token ratio, not surface variety"
  Policies:   grammar.evidence_required → True (was False)
              lexical.specificity → 4 (was 2)

Do you approve these changes? (approve / reject [with critique])
```

### 4. Process the Human Decision

**Approved:**
```
approve_proposal(session_id=session_id, iteration=iteration)
```
This updates `SessionState` with the new policies and strategy, increments the iteration, and marks the proposal `approved`.

**Rejected:**
```
reject_proposal(session_id=session_id, iteration=iteration, critique="<human's critique>")
```
This marks the proposal `rejected` and stores the critique. Then either:
- Re-run `run_chief_examiner` with the critique passed as additional context (the tool will incorporate it into the next diagnostic)
- Or manually adjust the proposal yourself based on the critique and use `approve_proposal` with the amended state

### 5. Confirm State Updated

After approval, verify:
```
state = load_state(drive_path, session_id)
state.iteration  # should be N+1
state.policies   # should reflect approved changes
state.strategy_guidance  # should reflect approved strategy
```

## Red Flags — Stop and Investigate

- Proposal changes more than 3 things at once — too many variables to attribute improvement next iteration; push back and ask for a focused change
- All four components flagged with the same issue — likely a global prompt problem (e.g. LLM refusing to penalise), not component-specific; the proposed fix should be global too
- `diagnostic_report.metric_summary` disagrees with your `interpret-metrics` brief by > 0.05 on any metric — stale data in the diagnostic
- Human approves without reading — prompt them to confirm they read `per_component_issues` before accepting

## Quick Reference

```
Order:  run_chief_examiner → read DiagnosticReport → present to human
        → approve_proposal or reject_proposal(critique)

After approval:  state.iteration == N+1, policies and strategy updated
After rejection: critique stored, re-run or manually amend before next pass

Never auto-approve. Never skip the human gate.
```
