# Internal Baseline Replication Before Session 1

The session history needs a baseline (Session 0) representing the performance this system aims to improve upon. The question is whether Session 0 uses metrics cited directly from Nguyen et al. (2026) or metrics from an internally-run replication.

We replicate the relevant Nguyen et al. modeling techniques ourselves on the same dataset, record the results, and write them as Session 0.

**Why:** Citing the paper's numbers directly introduces uncontrolled variables — hardware, data version, preprocessing differences, random seeds. Any performance gap between Session 0 and later Sessions would then be partly attributable to those differences rather than to this system's improvements. Running the replication internally ensures the comparison is apples-to-apples: same data split, same evaluation metrics (accuracy, adjacent accuracy, QWK, SMD), same hardware where possible.

**Consequences:** A baseline replication phase precedes Session 1. The replication is not part of the agentic loop — it is a one-time setup step that produces the Session 0 record. The Nguyen et al. paper's published numbers remain available as a secondary reference but are not the primary benchmark.
