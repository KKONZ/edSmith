from __future__ import annotations

import pandas as pd
import language_tool_python

from edsmith.tools import ToolResult
from edsmith.tools.aoa import _aoa_entry

_tool: language_tool_python.LanguageTool | None = None


def _get_tool() -> language_tool_python.LanguageTool:
    global _tool
    if _tool is None:
        _tool = language_tool_python.LanguageTool("en-US")
    return _tool


def grammar_check(text: str) -> ToolResult:
    matches = _get_tool().check(text)

    details = []
    for m in matches:
        span = text[m.offset : m.offset + m.errorLength].strip()
        # for multi-word spans, take the first token for AoA lookup
        word = span.lower().split()[0] if span else ""
        entry = _aoa_entry(word) if word else None
        details.append({
            "message": m.message,
            "word": span,
            "offset": m.offset,
            "length": m.errorLength,
            "replacements": m.replacements[:3],
            "rule_id": m.ruleId,
            "aoa": entry["aoa"] if entry else None,
            "nsyll": entry["nsyll"] if entry else None,
            "freq_pm": entry["freq_pm"] if entry else None,
        })

    count = len(matches)
    aoa_vals = [d["aoa"] for d in details if d["aoa"] is not None]

    stats: dict[str, float] = {"error_count": float(count)}
    if aoa_vals:
        aoa_s = pd.Series(aoa_vals)
        stats["mean_aoa_of_errors"] = aoa_s.mean()
        stats["std_aoa_of_errors"] = aoa_s.std()
        stats["pct_errors_basic"] = float((aoa_s < 7).mean() * 100)
        stats["pct_errors_advanced"] = float((aoa_s >= 10).mean() * 100)

    return ToolResult(
        tool="grammar",
        count=count,
        stats=stats,
        details=details,
        summary=f"{count} grammar/spelling issue(s) found",
    )
