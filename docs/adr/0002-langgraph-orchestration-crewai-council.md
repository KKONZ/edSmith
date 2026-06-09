# LangGraph for Session Orchestration, CrewAI for LLM-Council

The orchestration layer needs to handle an async training step (Colab GPU), stateful progression across iterations, and multi-agent composition for Feedback generation. Two frameworks were evaluated alongside a custom orchestration approach.

We use LangGraph for the outer Session/Iteration loop and CrewAI for the LLM-Council within Phase 1.

**Why LangGraph:** Its built-in checkpointing saves full graph state at each node. This directly solves the Colab lifecycle problem — the graph pauses after Phase 1 writes the augmented dataset to Google Drive, Colab runs fine-tuning and returns metrics, and the graph resumes exactly where it left off without custom state management. The conditional edge structure also maps naturally to the reflection escalation (simple → beam search → MCTS/LAST) as later iterations accumulate episodic memory.

**Why CrewAI for the council:** The Generator/Critic/Chair role pattern is a natural fit for CrewAI's crew abstraction. It runs as a subgraph called from within the LangGraph Phase 1 node and is bypassed entirely when LLM-Council is disabled.

**Consequences:** Augmented datasets and the vector store are persisted to Google Drive between phases, as Google Drive is natively mountable in Colab without additional infrastructure.

**Considered alternatives:** Pure LangGraph for everything (viable but verbose for the council role pattern), pure CrewAI (lacks checkpointing, poorly suited to the sequential stateful loop), custom orchestration (unnecessary complexity given both frameworks are maintained and well-documented).
