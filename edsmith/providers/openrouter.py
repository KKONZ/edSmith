from __future__ import annotations

import os

from openai import AsyncOpenAI, OpenAI

from edsmith.providers.base import CompletionResponse, LLMProvider, Message

_BASE_URL = "https://openrouter.ai/api/v1"


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
        max_tokens: int = 2048,
    ) -> CompletionResponse:
        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self._parse(response)

    async def acomplete(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> CompletionResponse:
        response = await self._async_client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self._parse(response)

    @staticmethod
    def _parse(response) -> CompletionResponse:
        choice = response.choices[0]
        usage = response.usage
        content = choice.message.content or ""
        if not content:
            finish = getattr(choice, "finish_reason", "unknown")
            raise RuntimeError(
                f"Model returned empty content (finish_reason={finish!r}, model={response.model!r}). "
                "Check model ID, rate limits, and OpenRouter dashboard."
            )
        return CompletionResponse(
            content=content,
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )
