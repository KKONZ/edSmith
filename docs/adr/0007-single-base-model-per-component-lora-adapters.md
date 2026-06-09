# Single Base Model with Per-Component LoRA Adapters

Four IELTS Components require separate scoring models. The question is whether to train fully separate models, a single multi-task model, or a shared base with per-Component adapters.

We use a single Qwen3 base model with one LoRA adapter per Component (4 adapters total). The base model is shared; adapters are swapped at inference time to get per-Component predictions.

**Why:** The shared base learns general essay-scoring representations that benefit all Components. Per-Component adapters provide specialisation without 4× training cost. Adapter files are lightweight relative to the full model. This fits naturally with Unsloth's LoRA setup and `AutoModelForSequenceClassification`.

**Considered alternatives:**
- Fully separate models: clean separation but 4× training cost per iteration and 4× storage — not justified given the shared domain.
- Single multi-task model with joint CORN heads: elegant but requires non-standard architecture that works against `AutoModelForSequenceClassification`; adds complexity before the simpler approach has been validated.
