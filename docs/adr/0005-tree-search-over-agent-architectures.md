# Tree Search Over Agent Architectures

The Feedback generation step requires an agent pattern — the question is whether to fix that pattern (e.g., always use a council) or treat it as a search variable.

The LAST/MCTS tree search's action space includes the full Feedback generation architecture. No pattern is assumed superior — the system discovers empirically which works best. Known starting patterns include simple ReAct (single agent, no council), full LLM-Council (generator + critic(s) + chair), and hybrids (e.g., ReAct generation with council critique). New patterns can be added and tested without changing the search mechanism.

**Why:** Different architectures may outperform others depending on dataset size, iteration phase, model choices, or Feedback granularity. Hardcoding an architecture assumes the answer to an empirical question the system is designed to resolve. Making architecture a search variable means the system can discover this — including non-obvious configurations like council-without-chair or multi-round critique.

**Consequences:** Each episodic memory Session record must capture the full architecture configuration under a structured `architecture` field (pattern type + all role-specific settings). The `action_taken` field must be expressive enough to describe both architectural transitions ("switched from ReAct to council") and parameter changes ("tightened generator prompt"). Architectural patterns are treated as first-class search dimensions alongside prompt content, k, Feedback granularity, and model selection.
