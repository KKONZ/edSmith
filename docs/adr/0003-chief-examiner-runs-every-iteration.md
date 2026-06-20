# Chief Examiner Runs After Every Iteration

The Chief Examiner runs after every Scorer evaluation, unconditionally. It is not triggered by performance thresholds, not skipped when metrics improve, and not bypassed after the first iteration.

**Why:** Every iteration produces new Feedback data and new Scorer performance numbers. The Chief Examiner uses both to produce a targeted proposal. Skipping a diagnostic pass because metrics improved would discard information about why they improved — which matters for avoiding regression in the next iteration. A fixed-threshold trigger would require per-dataset tuning and would activate late when the training sample is small and metrics are noisy.

**How it works:** After Cell 3 writes `metrics_iter{N}.json`, the orchestrating agent calls `run_chief_examiner`. The Chief Examiner receives the full iteration history (all prior approved proposals and their metrics) plus a sample of the current training Feedback sorted by score-band divergence. It produces a `DiagnosticReport` and `HumanReviewProposal`. The human approves or rejects. If rejected, the critique is stored and the Chief Examiner is called again in the same iteration with the critique injected. There is no skip condition and no minimum data threshold.

**Considered alternative — trigger only when validation accuracy declines:** Rejected because improving accuracy can still mask deteriorating diagnostic patterns (over-prediction creeping up, one component improving at the expense of another). The cost of the Chief Examiner call is negligible relative to the GPU training step, so there is no reason to skip it.
