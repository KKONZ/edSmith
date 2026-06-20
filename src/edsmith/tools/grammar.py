from __future__ import annotations

import language_tool_python

from edsmith.tools import ToolResult

_tool: language_tool_python.LanguageTool | None = None


def _get_tool() -> language_tool_python.LanguageTool:
    global _tool
    if _tool is None:
        _tool = language_tool_python.LanguageTool("en-US")
    return _tool


def grammar_check(text: str) -> ToolResult:
    matches = _get_tool().check(text)
    details = [
        {
            "message": m.message,
            "offset": m.offset,
            "length": m.errorLength,
            "replacements": m.replacements[:3],
            "rule_id": m.ruleId,
        }
        for m in matches
    ]
    count = len(matches)
    return ToolResult(
        tool="grammar",
        count=count,
        details=details,
        summary=f"{count} grammar/spelling issue(s) found",
    )
