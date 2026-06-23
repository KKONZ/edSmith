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

_ALL = {
    "aoa_stats": _AOA,
    "grammar_check": _GRAMMAR,
    "complexity_stats": _COMPLEXITY,
    "discourse_analysis": _DISCOURSE,
}


def get_tool_definitions(strategy: StrategyGuidance) -> list[dict]:
    """Return the subset of tool definitions enabled by the strategy."""
    tools = []
    if strategy.use_aoa:
        tools.append(_AOA)
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
            result = compute_aoa_stats(text)
        elif name == "grammar_check":
            from edsmith.tools.grammar import grammar_check
            result = grammar_check(text)
        elif name == "complexity_stats":
            from edsmith.tools.complexity import complexity_stats
            result = complexity_stats(text)
        elif name == "discourse_analysis":
            from edsmith.tools.discourse import discourse_analysis
            result = discourse_analysis(text)
        else:
            return json.dumps({"error": f"Unknown tool: {name}"})

        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})
