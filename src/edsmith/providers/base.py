from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string from the API


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None   # set on assistant turn that made calls
    tool_call_id: str | None = None            # set on tool result turns


@dataclass
class CompletionResponse:
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMProvider(ABC):
    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        enable_thinking: bool = False,
        tools: list[dict] | None = None,
    ) -> CompletionResponse: ...

    @abstractmethod
    async def acomplete(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        enable_thinking: bool = False,
        tools: list[dict] | None = None,
    ) -> CompletionResponse: ...
