import pytest

from edsmith.providers.base import CompletionResponse, LLMProvider


class StubProvider(LLMProvider):
    """Synchronous + async test double. Call .set(content) to configure the response."""

    def __init__(
        self,
        content: str = "",
        input_tokens: int = 10,
        output_tokens: int = 20,
    ) -> None:
        self._content = content
        self._input = input_tokens
        self._output = output_tokens

    def set(self, content: str) -> None:
        self._content = content

    def complete(self, messages, model, temperature=0.7, max_tokens=2048):
        return CompletionResponse(
            content=self._content,
            model=model,
            input_tokens=self._input,
            output_tokens=self._output,
        )

    async def acomplete(self, messages, model, temperature=0.7, max_tokens=2048):
        return self.complete(messages, model, temperature, max_tokens)


@pytest.fixture
def stub_provider():
    return StubProvider()
