# Human Gate via Orchestration Conversation

After each Scorer evaluation, the Chief Examiner produces a `DiagnosticReport` and `HumanReviewProposal`. Before any policy or strategy change is applied, a human must explicitly approve or reject the proposal. This gate is implemented as a natural pause in the orchestration conversation — not as a timer, a polling loop, or an automated threshold check.

**Why:** The diagnostic step is where the system decides what to change about how it generates Feedback. Getting this wrong costs an entire training iteration (~hours on a GPU). A human reading the `DiagnosticReport` catches misdiagnoses — cases where the Chief Examiner correctly identifies a metric problem but incorrectly attributes it to the Feedback rather than the data distribution, or proposes changes that are too broad to attribute causally. The conversation pause is zero-cost (the AI agent is already the orchestrator) and adds meaningful oversight at the highest-leverage decision point in the loop.

**How it works:** `run_chief_examiner` returns the proposal JSON. The orchestrating agent formats it for the human: diagnostic summary, per-component issues, and proposed changes. The human types `approve` or `reject [critique]`. The agent then calls `approve_proposal` or `reject_proposal(critique)`. If rejected, `run_chief_examiner` is called again with the stored critique, which is injected into the diagnostic prompt so the Chief Examiner's next proposal directly addresses the human's objection.

**What the human is not asked to do:** The human does not write the new policies or strategy — they only judge whether the Chief Examiner's proposal is reasonable. This keeps the gate lightweight. The human's role is a quality check, not authorship.

**Considered alternative — auto-approve when accuracy improves by more than a threshold:** Rejected because accuracy can improve for the wrong reason (the Scorer gaming a narrow distribution), and the threshold would need to be tuned per dataset. The conversation gate costs nothing and preserves full human agency over the search direction without requiring upfront calibration.
