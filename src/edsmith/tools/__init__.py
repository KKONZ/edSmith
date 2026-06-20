from typing import Any, TypedDict


class ToolResult(TypedDict):
    tool: str
    count: int
    stats: dict[str, float]   # aggregate statistics (mean, std, kurtosis, skew, etc.)
    details: list[dict[str, Any]]
    summary: str
