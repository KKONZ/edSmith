# Data Availability Triggers Reflection Mode Escalation

The reflection stage escalates through three modes across iterations: simple reflection, beam search, and MCTS/LAST. The question is what triggers each transition.

Transitions are triggered by the number of episodic memory records available in the search tree, not by a fixed iteration schedule or performance plateau.

**Why:** Beam search requires sibling sessions (same parent, same tree depth) to compare — without them, there is nothing to search across. MCTS/LAST requires a populated tree to make UCB1 scoring meaningful. A fixed schedule risks activating advanced search before sufficient data exists, producing degenerate results. A performance plateau is a valid signal but adds complexity and may trigger prematurely due to noisy metrics on small samples.

**How it works:** The orchestrator checks the episodic memory tree before choosing a reflection mode. Beam search activates when at least two sibling sessions exist at the current depth. MCTS/LAST activates when the tree contains sufficient nodes for UCB1 to discriminate between branches. Both thresholds are configurable by the human operator at Session start.
