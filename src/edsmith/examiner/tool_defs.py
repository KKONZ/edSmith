"""OpenAI-format tool definitions for linguistic analysis tools + executor."""
from __future__ import annotations

import json

from edsmith.config.session import StrategyGuidance

_AOA = {
    "type": "function",
    "function": {
        "name": "aoa_stats",
        "description": (
            "Compute age-of-acquisition (AoA) vocabulary statistics for a piece of text. "
            "Returns per-word AoA scores and summary stats: mean AoA, percentage of basic "
            "(AoA < 7) and advanced (AoA ≥ 10) vocabulary. Call this on the full essay or "
            "on specific sentences/paragraphs to compare vocabulary sophistication across sections."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to analyse."},
            },
            "required": ["text"],
        },
    },
}

_GRAMMAR = {
    "type": "function",
    "function": {
        "name": "grammar_check",
        "description": (
            "Check a piece of text for grammar and spelling errors using LanguageTool. "
            "Returns total error count, per-error details (message, word, rule ID, AoA of "
            "the error word), and aggregate stats. Useful for pinpointing specific error "
            "types and whether errors cluster in a particular part of the essay."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to check."},
            },
            "required": ["text"],
        },
    },
}

_COMPLEXITY = {
    "type": "function",
    "function": {
        "name": "complexity_stats",
        "description": (
            "Compute syntactic complexity statistics: mean sentence length, clause density, "
            "subordination ratio, and other structural metrics. Useful for assessing "
            "Grammatical Range — whether the writer uses a variety of complex structures."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to analyse."},
            },
            "required": ["text"],
        },
    },
}

_DISCOURSE = {
    "type": "function",
    "function": {
        "name": "discourse_analysis",
        "description": (
            "Analyse discourse structure and cohesive devices: transition words, "
            "referencing patterns, paragraph organisation. Useful for assessing "
            "Coherence and Cohesion."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to analyse."},
            },
            "required": ["text"],
        },
    },
}

_AOA_LOOKUP = {
    "type": "function",
    "function": {
        "name": "aoa_lookup",
        "description": (
            "Look up age-of-acquisition (AoA), syllable count, and frequency for a specific "
            "list of words. Use this when you want to check whether particular words the "
            "writer chose are basic or advanced — e.g. after spotting an interesting "
            "collocation or an unusually sophisticated/simple word choice."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "words": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of individual words to look up.",
                },
            },
            "required": ["words"],
        },
    },
}

_ALL = {
    "aoa_stats": _AOA,
    "aoa_lookup": _AOA_LOOKUP,
    "grammar_check": _GRAMMAR,
    "complexity_stats": _COMPLEXITY,
    "discourse_analysis": _DISCOURSE,
}


def get_tool_definitions(strategy: StrategyGuidance) -> list[dict]:
    """Return the subset of tool definitions enabled by the strategy."""
    tools = []
    if strategy.use_aoa:
        tools.append(_AOA)
        tools.append(_AOA_LOOKUP)
    if strategy.use_grammar:
        tools.append(_GRAMMAR)
    if strategy.use_complexity:
        tools.append(_COMPLEXITY)
    if strategy.use_discourse:
        tools.append(_DISCOURSE)
    return tools


def execute_tool(name: str, arguments: str) -> str:
    """Execute a tool call and return the result as a JSON string."""
    try:
        args = json.loads(arguments)
        text = args.get("text", "")

        if name == "aoa_stats":
            from edsmith.tools.aoa import compute_aoa_stats
            result = dict(compute_aoa_stats(text))
        elif name == "aoa_lookup":
            from edsmith.tools.aoa import aoa_lookup
            words = args.get("words", [])
            if not isinstance(words, list):
                words = [words] if words else []
            result = dict(aoa_lookup(words))
        elif name == "grammar_check":
            from edsmith.tools.grammar import grammar_check
            result = dict(grammar_check(text))
        elif name == "complexity_stats":
            from edsmith.tools.complexity import complexity_stats
            result = dict(complexity_stats(text))
        elif name == "discourse_analysis":
            from edsmith.tools.discourse import discourse_analysis
            result = dict(discourse_analysis(text))
        elif name == "pos_tag":
            return json.dumps({"error": "pos_tag tool has been removed"})
        else:
            return json.dumps({"error": f"Unknown tool: {name}"})

        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})
