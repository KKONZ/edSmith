# Structured Prompt Policy as the Tree Search Interface

The tree search needs to modify Feedback generation configuration across iterations. The question is whether the agent operates on raw prompt text or a structured abstraction.

We use a structured prompt policy as the primary search interface. Each Component has its own policy with typed fields (e.g., specificity level, evidence required, feedback granularity). A free-text `additional_instructions` field is available alongside the structured fields for nuance the schema cannot capture.

**Why:** Raw prompt text gives the agent an unbounded search space that is difficult to search systematically, hard to compare across sessions, and easy to overfit. Structured fields give the tree search a finite, well-defined action space where changes between sessions are explicit and comparable. The `additional_instructions` escape hatch preserves flexibility without making it the default.

**How it works:** The tree search modifies structured policy fields first. The `additional_instructions` field becomes relevant once the tree has enough episodic data to reason about subtler changes that the structured fields cannot express. Policy objects are versioned per session in episodic memory.
