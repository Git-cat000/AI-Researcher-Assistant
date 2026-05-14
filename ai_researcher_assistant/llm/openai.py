"""OpenAI-compatible chat adapter.

This adapter supports OpenAI and OpenAI-compatible providers such as Azure
OpenAI, DeepSeek, OpenRouter, SiliconFlow, local gateways, and compatible
enterprise endpoints through `base_url`.
"""

from collections.abc import AsyncIterator
import os
from typing import Any

from ai_researcher_assistant.core.exceptions import LLMError
from ai_researcher_assistant.llm.base import BaseLLM, LLMResponse


class OpenAILLM(BaseLLM):
    """LLM adapter for OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMError("OpenAI-compatible API key not found. Set OPENAI_API_KEY or pass api_key.")

        try:
            import openai
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise LLMError("OpenAI SDK is not installed. Run `pip install openai`.") from exc

        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", 1.0),
                stop=kwargs.get("stop"),
            )
            return self._parse_response(response)
        except Exception as exc:
            raise LLMError(f"OpenAI-compatible API error: {exc}") from exc

    async def agenerate(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", 1.0),
                stop=kwargs.get("stop"),
            )
            return self._parse_response(response)
        except Exception as exc:
            raise LLMError(f"OpenAI-compatible async API error: {exc}") from exc

    async def stream_generate(self, messages: list[dict[str, str]], **kwargs: Any) -> AsyncIterator[str]:
        try:
            stream = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", 1.0),
                stop=kwargs.get("stop"),
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as exc:
            raise LLMError(f"OpenAI-compatible streaming error: {exc}") from exc

    def _parse_response(self, response: Any) -> LLMResponse:
        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            if response.usage
            else None,
            finish_reason=choice.finish_reason,
            raw_response=response,
        )
