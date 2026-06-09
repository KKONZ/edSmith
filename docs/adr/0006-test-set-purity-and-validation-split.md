# Test Set Purity and Validation Split

The reflection stage needs individual-record signal (essays, Feedback, misses) to reason about failure modes. The risk is that allowing the reflection stage to observe individual test set records lets the system implicitly tune toward specific test examples across iterations, invalidating the benchmark.

The test set is strictly off-limits to the reflection stage at the record level. Only summary statistics derived from test set evaluation (accuracy, adjacent accuracy, QWK, SMD) are passed to the reflection stage. The reflection stage inspects individual records only from training data.

Two record-level signals are available to the reflection stage:

1. **Validation split** — a fixed subset of the designated training sample, held out from fine-tuning, available for individual-record inspection every iteration. Provides a stable window into the data consistent across iterations. Carved from the Session sample at a configurable ratio before Session 1.

2. **Training misses** — wrongly predicted training examples from the current iteration, grouped by failure pattern (miss clustering). Provides the sharpest signal about current failure modes. Dynamic — the set changes as the model improves.

The validation split is fixed for the duration of the Session alongside the training sample, consistent with the principle that random effects are eliminated by holding splits constant.

**Why:** Allowing the reflection stage to observe individual test records risks implicit overfitting to the benchmark — the system learns to handle specific test essays rather than generalising. This would make metric improvements across iterations uninterpretable. The constraint is non-obvious because it restricts a capability (rich reflection signal) that would appear helpful in isolation.
