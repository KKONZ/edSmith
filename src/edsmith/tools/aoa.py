from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from edsmith.tools import ToolResult

_DATA_PATH = Path(__file__).parent / "data" / "aoa.parquet"
_lookup: dict[str, dict] | None = None


def _get_lookup() -> dict[str, dict]:
    global _lookup
    if _lookup is None:
        df = pd.read_parquet(_DATA_PATH)
        _lookup = df.set_index("word").to_dict(orient="index")
    return _lookup


def _aoa_entry(word: str) -> dict | None:
    return _get_lookup().get(word.lower())


def aoa_lookup(words: list[str]) -> ToolResult:
    """Look up AoA, syllable count, and frequency for a specific list of words.

    If a word is not found, it may be a misspelling. Use your language understanding
    to infer what the writer most likely intended, then call this tool again with the
    corrected spelling to get its AoA and vocabulary level.
    """
    lookup = _get_lookup()
    details = []
    missing = []

    for w in words:
        entry = lookup.get(w.lower())
        if entry is not None:
            details.append({"word": w.lower(), **entry})
        else:
            missing.append(w.lower())

    aoa_vals = [d["aoa"] for d in details]
    stats: dict[str, float] = {"found": len(details), "missing": len(missing)}
    if aoa_vals:
        import statistics
        stats["aoa_mean"] = statistics.mean(aoa_vals)
        stats["aoa_min"] = min(aoa_vals)
        stats["aoa_max"] = max(aoa_vals)

    summary_parts = [f"{len(details)}/{len(words)} words found in AoA lookup"]
    if missing:
        summary_parts.append(f"not found (possible misspellings — infer intended word and retry): {', '.join(missing)}")
    if aoa_vals:
        summary_parts.append(f"mean AoA {stats['aoa_mean']:.2f} (range {stats['aoa_min']:.1f}–{stats['aoa_max']:.1f})")

    return ToolResult(
        tool="aoa_lookup",
        count=len(details),
        stats=stats,
        details=details,
        summary="; ".join(summary_parts),
    )


def compute_aoa_stats(text: str) -> ToolResult:
    if not text.strip():
        return ToolResult(
            tool="aoa", count=0, stats={}, details=[], summary="No text provided"
        )

    lookup = _get_lookup()
    tokens = re.findall(r"[a-z]+", text.lower())

    details = []
    for token in tokens:
        entry = lookup.get(token)
        if entry is not None:
            details.append({"word": token, **entry})

    if not details:
        return ToolResult(
            tool="aoa",
            count=0,
            stats={"coverage": 0.0},
            details=[],
            summary=f"0/{len(tokens)} tokens found in AoA lookup",
        )

    aoa_s = pd.Series([d["aoa"] for d in details])
    nsyll_s = pd.Series([d["nsyll"] for d in details if d.get("nsyll") is not None])
    freq_s = pd.Series([d["freq_pm"] for d in details if d.get("freq_pm") is not None])
    coverage = len(details) / len(tokens) * 100

    stats: dict[str, float] = {
        "coverage": coverage,
        "aoa_mean": aoa_s.mean(),
        "aoa_std": aoa_s.std(),
        "aoa_median": aoa_s.median(),
        "aoa_skew": aoa_s.skew(),
        "aoa_kurtosis": aoa_s.kurtosis(),
        "aoa_min": aoa_s.min(),
        "aoa_max": aoa_s.max(),
        "pct_early": float((aoa_s < 7).mean() * 100),
        "pct_late": float((aoa_s >= 10).mean() * 100),
    }
    if not nsyll_s.empty:
        stats["nsyll_mean"] = nsyll_s.mean()
        stats["nsyll_std"] = nsyll_s.std()
    if not freq_s.empty:
        stats["freq_mean"] = freq_s.mean()
        stats["freq_std"] = freq_s.std()

    return ToolResult(
        tool="aoa",
        count=len(details),
        stats=stats,
        details=[],
        summary=(
            f"{len(details)}/{len(tokens)} tokens matched ({coverage:.0f}% coverage); "
            f"mean AoA {stats['aoa_mean']:.2f} ± {stats['aoa_std']:.2f}; "
            f"skew {stats['aoa_skew']:.2f}; kurtosis {stats['aoa_kurtosis']:.2f}; "
            f"{stats['pct_late']:.0f}% late-acquired (AoA ≥ 10)"
        ),
    )
