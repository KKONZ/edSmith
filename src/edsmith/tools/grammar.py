from __future__ import annotations

from pathlib import Path

import pandas as pd
import language_tool_python

from edsmith.tools import ToolResult
from edsmith.tools.aoa import _aoa_entry

_tool: language_tool_python.LanguageTool | None = None


def _get_tool() -> language_tool_python.LanguageTool:
    global _tool
    if _tool is None:
        import os
        java_home = os.environ.get("JAVA_HOME")
        if java_home:
            bin_dir = str(Path(java_home) / "bin")
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        _tool = language_tool_python.LanguageTool("en-US")
    return _tool


def grammar_check(text: str) -> ToolResult:
    matches = _get_tool().check(text)

    details = []
    for m in matches:
        span = text[m.offset : m.offset + m.error_length].strip()
        word = span.lower().split()[0] if span else ""
        entry = _aoa_entry(word) if word else None
        # If AoA not found (e.g. British spelling), try single replacement (e.g. favour→favor)
        if entry is None and len(m.replacements) == 1:
            entry = _aoa_entry(m.replacements[0].lower().split()[0])
        details.append({
            "message": m.message,
            "word": span,
            "replacements": m.replacements[:3],
            "aoa": entry["aoa"] if entry else None,
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
