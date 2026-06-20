# Deterministic Tool-Based Evaluation Over LLM-as-Judge

The Chief Examiner needs a signal to judge whether changes to Feedback generation (Prompt Policies, Strategy Guidance) improved outcomes. We evaluated two approaches: LLM-as-judge (an LLM rates Feedback quality) and deterministic metrics computed against a held-out test set.

We use deterministic metrics — accuracy, adjacent accuracy, quadratic weighted kappa (QWK), and standardized mean difference (SMD) — computed via tool calls against a held-out test set. Training uses a small number of steps per turn (via Unsloth sequence classification) rather than full fine-tuning.

**Why:** LLM-as-judge creates a circular optimization loop where the agent could learn to satisfy its evaluator rather than improve genuine scoring performance. Deterministic metrics ground the signal in actual model behaviour on unseen data, making reward hacking significantly harder. QWK is the established standard for ordinal scoring tasks (including IELTS); adjacent accuracy and SMD complement it for fine-grained diagnosis.

**Considered alternatives:** A staged approach using LLM-as-judge as a fast pre-filter before committing to a full training run was considered. Rejected for now because the per-turn training cost (few steps on a small model) is low enough that the fast proxy is not needed. May revisit if turn cost grows.
