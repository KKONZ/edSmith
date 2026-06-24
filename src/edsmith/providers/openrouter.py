from __future__ import annotations

import logging
import os

from openai import AsyncOpenAI, OpenAI

from edsmith.providers.base import CompletionResponse, LLMProvider, Message, ToolCall

_BASE_URL = "https://openrouter.ai/api/v1"
logger = logging.getLogger("edsmith.providers.openrouter")


def _to_api_message(m: Message) -> dict:
    msg: dict = {"role": m.role}
    if m.content is not None:
        msg["content"] = m.content
    if m.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in m.tool_calls
        ]
    if m.tool_call_id:
        msg["tool_call_id"] = m.tool_call_id
    return msg


class OpenRouterProvider(LLMProvider):
    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY not set")
        self._client = OpenAI(api_key=key, base_url=_BASE_URL, max_retries=5)
        self._async_client = AsyncOpenAI(api_key=key, base_url=_BASE_URL, max_retries=5)

    def complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        enable_thinking: bool = False,
        tools: list[dict] | None = None,
    ) -> CompletionResponse:
        extra: dict = {}
        if not enable_thinking:
            extra["reasoning"] = {"exclude": True}
        kwargs: dict = dict(
            model=model,
            messages=[_to_api_message(m) for m in messages],
            temperature=temperature,
            extra_body=extra if extra else None,
        )
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = tools
        response = self._client.chat.completions.create(**kwargs)
        return self._parse(response)

    async def acomplete(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        enable_thinking: bool = False,
        tools: list[dict] | None = None,
    ) -> CompletionResponse:
        extra: dict = {}
        if not enable_thinking:
            extra["reasoning"] = {"exclude": True}
        kwargs: dict = dict(
            model=model,
            messages=[_to_api_message(m) for m in messages],
            temperature=temperature,
            extra_body=extra if extra else None,
        )
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = tools
        response = await self._async_client.chat.completions.create(**kwargs)
        return self._parse(response)

    @staticmethod
    def _parse(response) -> CompletionResponse:
        if not response.choices:
            raise RuntimeError(
                f"OpenRouter returned no choices (model={response.model!r}). "
                "Possible causes: null/empty content in the request, rate limit, or provider error."
            )
        choice = response.choices[0]
        usage = response.usage

        raw_tool_calls: list[ToolCall] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                raw_tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                ))

        content = choice.message.content or ""
        if not content and not raw_tool_calls:
            finish = getattr(choice, "finish_reason", "unknown")
            raise RuntimeError(
                f"Model returned empty content (finish_reason={finish!r}, model={response.model!r}). "
                "Check model ID, rate limits, and OpenRouter dashboard."
            )

        result = CompletionResponse(
            content=content,
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            tool_calls=raw_tool_calls,
        )
        logger.debug(
            "api_call model=%s in=%d out=%d finish=%s tool_calls=%d",
            response.model,
            result.input_tokens,
            result.output_tokens,
            choice.finish_reason,
            len(raw_tool_calls),
        )
        return result
