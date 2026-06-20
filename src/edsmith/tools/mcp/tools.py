from __future__ import annotations

from fastmcp import FastMCP

from edsmith.tools.aoa import compute_aoa_stats
from edsmith.tools.complexity import complexity_stats as _complexity_stats
from edsmith.tools.discourse import discourse_analysis as _discourse_analysis
from edsmith.tools.grammar import grammar_check as _grammar_check


def register_grammar_check(app: FastMCP):
    @app.tool(
        title="Grammar Check",
        description=(
            "Check text for grammar and spelling errors. Each flagged error is "
            "cross-referenced with its Age-of-Acquisition score so callers can "
            "distinguish errors on basic vs. advanced vocabulary. "
            "Requires the [tools] optional dependencies."
        ),
    )
    def grammar_check(text: str) -> dict:
        return dict(_grammar_check(text))

    return grammar_check


def register_aoa_stats(app: FastMCP):
    @app.tool(
        title="Age-of-Acquisition Statistics",
        description=(
            "Compute Age-of-Acquisition statistics for vocabulary in the text. "
            "Returns distribution stats (mean, std, skew, kurtosis, % early/late "
            "acquired), syllable and frequency stats, and per-word detail entries."
        ),
    )
    def aoa_stats(text: str) -> dict:
        return dict(compute_aoa_stats(text))

    return aoa_stats


def register_complexity_stats(app: FastMCP):
    @app.tool(
        title="Syntactic Complexity Statistics",
        description=(
            "Compute syntactic complexity statistics for the text. Includes "
            "per-sentence dependency depth, passive/subordinate/nominalization "
            "ratios, type-token ratio, and per-sentence AoA cross-reference. "
            "Requires the [tools] optional dependencies."
        ),
    )
    def complexity_stats(text: str) -> dict:
        return dict(_complexity_stats(text))

    return complexity_stats


def register_discourse_analysis(app: FastMCP):
    @app.tool(
        title="Discourse Analysis",
        description=(
            "Analyse essay discourse structure. Segments into paragraphs and "
            "assigns introduction/body/conclusion roles. Detects transition words "
            "from a non-exhaustive example list categorised as additive, adversative, "
            "causal, sequential, exemplification, conclusion, and hedging. "
            "Cross-references with spaCy POS-detected connectives (SCONJ/CCONJ) to "
            "surface discourse markers not in the wordlist. Reports pronoun ratio and "
            "cross-paragraph lexical repetition as cohesion signals. "
            "Requires the [tools] optional dependencies."
        ),
    )
    def discourse_analysis(text: str) -> dict:
        return dict(_discourse_analysis(text))

    return discourse_analysis
