# Linguistic Feature Tools

Four linguistic analysis tools are exposed as MCP tools in `src/edsmith/tools/`: `grammar_check`, `aoa_stats`, `complexity_stats`, and `discourse_analysis`. Each wraps a pure-Python domain implementation and is registered on the top-level FastMCP server via a `register_*(app: FastMCP)` function.

**grammar_check:** Uses `language_tool_python` to detect grammar and spelling errors. Each flagged error is cross-referenced with the Age-of-Acquisition score of the flagged word, so the Chief Examiner can distinguish errors on basic vocabulary (low AoA) from errors on advanced vocabulary (high AoA) — a meaningfully different diagnostic signal.

**aoa_stats:** Looks up each token in a locally-stored parquet (31k words from StephanAkkerman/English-Age-of-Acquisition on HuggingFace). Returns distribution statistics (mean, std, skew, kurtosis, % early/late acquired) alongside per-word details including syllable count, frequency, and part of speech. The full distribution — not just the mean — is the useful signal: a high-kurtosis AoA distribution indicates vocabulary clustered in a narrow acquisition band, while high skew indicates a systematic lean toward rare or common words.

**complexity_stats:** Uses spaCy to compute per-sentence dependency depth, passive voice ratio, subordinate clause ratio, nominalization ratio, and type-token ratio. Per-sentence AoA scores are cross-referenced via lemma lookup. NER is disabled on load since it is unused by any tool in this package.

**discourse_analysis:** Segments text into paragraphs and classifies each as introduction, body, or conclusion by position. Detects transition words from a categorised example list (additive, adversative, causal, sequential, exemplification, conclusion, hedging). The wordlist is intentionally non-exhaustive; spaCy POS tags (SCONJ, CCONJ) provide a cross-reference that surfaces connective words not in the list. The `wordlist_coverage_ratio` stat shows what fraction of POS-detected connectives the wordlist captured. Also reports pronoun ratio and cross-paragraph lexical repetition rate as substitution and cohesion signals.

**Why separate tools rather than one combined tool:** Each tool can be called independently by the Chief Examiner based on which signal is diagnostically relevant to the current iteration. Calling all four on every essay would be wasteful; the Chief Examiner's strategy guidance (`use_grammar`, `use_aoa`, `use_complexity`) controls which tools the Examiner invokes per component.

**Registration pattern:** Each domain's `mcp/__init__.py` exports `register_*` functions that accept an `app: FastMCP` argument and define the tool as an inner function decorated with `@app.tool()`. The top-level `src/edsmith/mcp/__main__.py` creates the single `FastMCP` instance and calls each `register_*` function. This avoids coupling domain modules to a shared mutable app object and keeps tool definitions co-located with their domain logic.

**Optional dependencies:** All four tools require the `[tools]` extras (`language-tool-python`, `spacy`, `en-core-web-sm`). Tests skip cleanly when these are not installed. The AoA parquet is committed to the repository and requires no network access at runtime.
