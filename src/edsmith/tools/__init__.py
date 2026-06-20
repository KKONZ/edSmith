from typing import Any, TypedDict


class ToolResult(TypedDict):
    tool: str
    count: int
    details: list[dict[str, Any]]
    summary: str
