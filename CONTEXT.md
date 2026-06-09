# edSmith

**Question**:
The writing prompt a student responds to with an Essay. Multiple Questions may share the same Rubric.
_Avoid_: Prompt, task

**Essay**:
The answer a student writes in response to a Question.
_Avoid_: Response, submission

**Rubric**:
The scoring criteria that defines the grade bands for each Component of a Question. A single Rubric may apply to multiple Questions.
_Avoid_: Criteria

**Scoring Guide**:
An optional set of worked examples — each containing an Essay, its Scores, and Feedback — used to illustrate how a Rubric is applied in practice. When available, may be used as few-shot examples during Feedback generation. Not always present.
_Avoid_: Rubric, examples (use Scoring Guide when referring to this artifact specifically)

**Component**:
A specific dimension of a Rubric against which a student's Essay is graded. A Question may have one or more Components; when multiple exist, each receives a separate Score. The IELTS dataset uses the word "criteria" for this concept — in this codebase, Component is always preferred.
_Avoid_: Criterion, criteria, dimension

**Score**:
The grade assigned to an Essay for a given Component, determined by the Rubric. An Essay receives one Score per Component.
_Avoid_: Grade, mark


**IELTS**:
The type of test a student is taking. This refers to the International English Language Testing System. The data this project is based on is from student essays and scores from IELTS tests.
_Avoid_:

**Validation Set**:
A fixed subset of the designated training sample held out from fine-tuning. Used by the reflection stage for individual-record inspection (essays, Feedback, misses). Carved from the Session sample before Session 1 at a configurable ratio and held constant for the Session's duration. Distinct from the test set, which the reflection stage may never observe at the record level.
_Avoid_: Dev set, hold-out set (use Validation Set)

**Baseline**:
An internal replication of the Nguyen et al. (2026) modeling techniques run on the same dataset before Session 1 begins. Results are recorded as Session 0 in the episodic memory tree and serve as the reference point against which all subsequent Sessions are measured.
_Avoid_: Control, reference model

**Scorer**:
The fine-tuned Qwen3 sequence classification model that predicts a Score for a given Component. A single base model with one LoRA adapter per Component. Trained during Phase 2 of each Iteration using augmented data from Phase 1.
_Avoid_: SML, classifier, fine-tuned model (use Scorer)

**Prompt Policy**:
A structured, per-Component configuration that controls how Feedback is generated during Phase 1. Contains typed fields (e.g., specificity, evidence required, feedback granularity) plus an optional free-text additional_instructions field. The tree search modifies policy fields between Iterations rather than operating on raw prompt text directly.
_Avoid_: Prompt template, prompt config (use Prompt Policy)

**Iteration**:
A single pass through Phase 1 (augmentation) and Phase 2 (fine-tuning and evaluation). Multiple iterations make up a Session. Within a Session, all iterations operate on the same fixed training sample to eliminate random effects.
_Avoid_: Turn, pass, round, experiment (use Iteration)

**Session**:
A configured loop run consisting of N iterations. The training sample, hold-out set, and council configuration are fixed at the start of a Session and held constant throughout.
_Avoid_: Run, experiment, loop (use Session)

**Feedback**:
A natural language explanation of why a student received a given score, addressable at the component level, the overall level, or both. Serves dual purpose: as a student-facing explanation of their score, and as chain-of-thought reasoning that the scorer model learns from during fine-tuning. Balancing human-readability against training signal quality is a core design tension; improving fine-tuning performance is the primary objective. The optimal granularity and formulation are determined empirically through agentic experimentation.
_Avoid_: Rationale, reasoning, explanation (use Feedback)