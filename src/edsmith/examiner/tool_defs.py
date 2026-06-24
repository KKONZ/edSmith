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

_POS = {
    "type": "function",
    "function": {
        "name": "pos_tag",
        "description": (
            "POS-tag a piece of text with spaCy. Returns per-token part-of-speech, "
            "fine-grained tag, syntactic dependency relation, and lemma. Call this on "
            "specific sentences to understand grammatical structure — e.g. to check "
            "whether the writer uses subordinate clauses, varied verb forms, or relies "
            "heavily on simple noun phrases."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to tag (sentence or paragraph)."},
            },
            "required": ["text"],
        },
    },
}

_ALL = {
    "aoa_stats": _AOA,
    "aoa_lookup": _AOA_LOOKUP,
    "grammar_check": _GRAMMAR,
    "complexity_stats": _COMPLEXITY,
    "discourse_analysis": _DISCOURSE,
    "pos_tag": _POS,
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
    if strategy.use_pos:
        tools.append(_POS)
    return tools


def _trim_result(result: dict, max_details: int = 10) -> dict:
    """Keep summary + stats but cap the details list to avoid context bloat."""
    trimmed = {k: v for k, v in result.items() if k != "details"}
    details = result.get("details", [])
    if len(details) > max_details:
        half = max_details // 2
        trimmed["details"] = details[:half] + details[-half:]
        trimmed["details_truncated"] = f"{len(details) - max_details} entries omitted (showing first/last {half})"
    else:
        trimmed["details"] = details
    return trimmed


def execute_tool(name: str, arguments: str) -> str:
    """Execute a tool call and return the result as a JSON string."""
    try:
        args = json.loads(arguments)
        text = args.get("text", "")

        if name == "aoa_stats":
            from edsmith.tools.aoa import compute_aoa_stats
            result = _trim_result(compute_aoa_stats(text))
        elif name == "aoa_lookup":
            from edsmith.tools.aoa import aoa_lookup
            words = args.get("words", [])
            if not isinstance(words, list):
                words = [words] if words else []
            result = _trim_result(aoa_lookup(words), max_details=20)
        elif name == "pos_tag":
            from edsmith.tools.pos import pos_tag
            result = _trim_result(pos_tag(text), max_details=20)
        elif name == "grammar_check":
            from edsmith.tools.grammar import grammar_check
            result = _trim_result(grammar_check(text), max_details=15)
        elif name == "complexity_stats":
            from edsmith.tools.complexity import complexity_stats
            result = _trim_result(complexity_stats(text), max_details=15)
        elif name == "discourse_analysis":
            from edsmith.tools.discourse import discourse_analysis
            result = _trim_result(discourse_analysis(text), max_details=10)
        else:
            return json.dumps({"error": f"Unknown tool: {name}"})

        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})
