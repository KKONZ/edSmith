# Examiner Agent

Generates per-component IELTS Feedback and Scores for all training Essays in one iteration. Called by the edsmith orchestrator after `approve_proposal` confirms the current `StrategyGuidance` and `PromptPolicy` values.

## Purpose

The Examiner produces the training signal that the Scorer learns from. Its job is to label every training Essay with a band Score and narrative Feedback for each of the four IELTS components. The Chief Examiner later diagnoses whether those labels are calibrated and coherent, and proposes changes to the strategy or policies for the next iteration.

The Examiner does not persist state between calls — it reads `SessionState` from disk on each invocation and writes one feedback parquet per iteration.

## Session Context

The Examiner is invoked once per iteration, after:
1. A `SessionState` exists on disk (created by `edsmith start-session` or a prior approved proposal).
2. The human (or auto-approve) has confirmed the `StrategyGuidance` and `PromptPolicy` values via `approve_proposal`.

Do not call `run_examiner_pass` before a `SessionState` exists for the session.

## MCP Tool

### `run_examiner_pass`

```
run_examiner_pass(session_id: str, iteration: int, concurrency: int = 4) -> ExaminerSummary
```

**What it reads:**
- `{drive_path}/sessions/{session_id}/state.json` — current `PromptPolicy` per component and `StrategyGuidance`
- `{drive_path}/sessions/{session_id}/data/train.parquet` — training Essays and their reference band Scores; downloaded from HuggingFace and saved on first call if absent

**What it writes:**
- `{drive_path}/sessions/{session_id}/feedback_iter{n}.parquet` — one row per (essay, component), columns: `question`, `essay`, `band`, `component`, `feedback_text`, `score`, `tag`

**`concurrency`**: maximum simultaneous LLM calls. Default 4 is safe for OpenRouter rate limits. Raise to 8–16 only if you have confirmed higher rate limit headroom.

**Return value — `ExaminerSummary`:**

| Field | Type | Meaning |
|---|---|---|
| `session_id` | str | Session identifier |
| `iteration` | int | Iteration number this pass covers |
| `essays_processed` | int | Unique essays that produced at least one feedback row |
| `essays_total` | int | Total essays in the training split |
| `components_covered` | int | Essays with all four components present (no partial failures) |
| `score_distributions` | dict | Per-component `{mean, std, count}` of predicted Scores |
| `warnings` | list[str] | One entry per essay that raised an exception during generation |
| `parquet_path` | str | Absolute path to the written feedback parquet |

## Interpreting the ExaminerSummary

**`essays_processed` < `essays_total`**: Some essays were skipped entirely — check `warnings`. Usually an API timeout or malformed response. If more than 5% of essays failed, re-run rather than proceeding to `run_chief_examiner`.

**`components_covered` < `essays_processed`**: Some essays have partial coverage (fewer than 4 components). The Chief Examiner can still diagnose from partial data, but note it in context when calling `run_chief_examiner`.

**`score_distributions`**: Compare across iterations. A collapsing standard deviation (std → 0) for a component means the Examiner is assigning near-uniform Scores — a sign of an under-specified `PromptPolicy` or an overly narrow `per_component_focus`. A mean drifting above 7 or below 4 for a component suggests calibration issues the Chief Examiner should flag.

**`warnings`**: Each string identifies the essay that failed and the exception. A single transient API error is ignorable. A repeating pattern (e.g., all failures on the same component) signals a prompt construction problem — inspect the policy for that component before proceeding.

## Strategy Guidance Fields

`StrategyGuidance` is set by the Chief Examiner and persisted in `SessionState`. The Examiner reads it on every call. These fields control what extra context is injected into each LLM prompt.

| Field | Default | Effect |
|---|---|---|
| `use_grammar` | `false` | Prepend grammar error summary from `language_tool_python` to the system prompt |
| `use_aoa` | `false` | Prepend vocabulary Age-of-Acquisition statistics (mean AoA, lexical diversity) |
| `use_complexity` | `false` | Prepend syntactic complexity metrics (mean sentence length, subordinate clause ratio, passive ratio) |
| `use_discourse` | `false` | Prepend discourse structure analysis (paragraph roles, transition word categories, cohesion signals) |
| `contrastive_anchoring` | `false` | Instruct the LLM to explicitly compare the essay to what a higher and lower band would look like for this component |
| `per_component_focus` | `{}` | Per-component free-text instruction appended after the standard strategy lines. Keys are component names (see below). An empty string for a component means no override. |

Enable linguistic tools selectively: each enabled tool adds latency and token cost. Start with `use_grammar: true` for `grammar` component feedback; add `use_complexity` and `use_discourse` if coherence Scores are poorly calibrated.

## Prompt Policy Fields

Each of the four components has its own `PromptPolicy` in `SessionState.policies`. The Examiner reads the policy keyed to each component.

| Field | Type | Effect on prompt |
|---|---|---|
| `specificity` | int 1–5 | Controls detail level: 1=brief overview, 5=highly detailed analysis |
| `evidence_required` | bool | When `true`, the LLM is instructed to cite specific evidence from the essay for every point |
| `feedback_granularity` | `"component"` \| `"overall"` \| `"both"` | `"both"` adds overall writing quality observations in addition to component-specific feedback |
| `additional_instructions` | str | Free-form text appended to the system prompt for this component; use for targeted corrections the Chief Examiner identified in the diagnostic |

## IELTS Components

| Key | Display name |
|---|---|
| `task_response` | Task Achievement |
| `coherence` | Coherence and Cohesion |
| `lexical` | Lexical Resource |
| `grammar` | Grammatical Range and Accuracy |

Use these exact keys in `per_component_focus` and when reading `score_distributions`.

## Typical Call Sequence

```
1. approve_proposal(session_id, iteration)          # confirms strategy + policies
2. run_examiner_pass(session_id, iteration)          # generate feedback parquet
3. [inspect ExaminerSummary — check warnings]
4. run_chief_examiner(session_id, iteration)         # diagnose feedback quality
```

If `run_examiner_pass` returns warnings covering more than 5% of essays, re-run before calling `run_chief_examiner`. If warnings persist, reduce `concurrency` to 2 and check `OPENROUTER_API_KEY` is set.
