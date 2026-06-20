# Structured PromptPolicy and StrategyGuidance as the Search Interface

Feedback generation is configured through two structured models persisted in `SessionState` on disk: `PromptPolicy` (one per IELTS component) and `StrategyGuidance` (one per session). The Chief Examiner proposes changes to both after each iteration; the human approves or rejects the proposal before changes are applied.

**Two layers, two concerns:**

`PromptPolicy` controls how the Examiner formats and styles its Feedback for a given component — specificity level, whether evidence must be cited, feedback granularity (component-only vs. overall), and a free-text `additional_instructions` escape hatch. These fields affect every essay in the component uniformly.

`StrategyGuidance` controls what information the Examiner collects before generating Feedback — which linguistic tool outputs to inject (grammar errors, AoA statistics, syntactic complexity, discourse structure), whether to use contrastive anchoring, and per-component focus instructions for this iteration. These fields affect the Examiner's information environment rather than its output format.

**Why structured fields over raw prompt text:** An unstructured prompt gives the Chief Examiner an unbounded action space that is hard to compare across iterations and easy to overfit to a specific training batch. Structured fields make each proposed change explicit and auditable — the human reviewing a proposal can see exactly which fields changed and by how much. The `additional_instructions` field provides flexibility for targeted corrections the schema cannot capture without making free-form editing the default.

**Why two models instead of one:** `PromptPolicy` changes are relatively fine-grained and component-specific. `StrategyGuidance` changes are session-wide and affect cost and latency (each enabled linguistic tool adds an extra analysis step per essay). Separating them lets the Chief Examiner and human reason about format changes and information changes independently, and makes the proposal easier to evaluate.

**How changes are applied:** `approve_proposal` writes the proposed `StrategyGuidance` and all four component `PromptPolicy` values to `SessionState` and increments the iteration counter. The next `run_examiner_pass` reads the updated state. If the human rejects, the critique is stored and the Chief Examiner generates a revised proposal in the same iteration — `SessionState` is not modified until an approval.

**Consequences:** Every accepted change is recorded as an approved `HumanReviewProposal` on disk, giving the Chief Examiner a complete history of what was tried and what the human said about each proposal. The iteration history is the primary mechanism for avoiding redundant proposals and building on prior progress.
